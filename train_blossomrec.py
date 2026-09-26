from __future__ import annotations

import itertools

import json

from pathlib import Path

import random

import time

import torch

import torch.nn.functional as F

from deltarec.utils import blossom_training as legacy

from deltarec.utils.io import atomic_write_json, sha256_file

from deltarec.utils.checkpoint import atomic_torch_save

from deltarec.utils.early_stopping import EarlyStoppingState

from deltarec.utils.accumulation import FuxiGradientAccumulation

from deltarec.utils.trajectory import SAMPLING_POLICY, TARGET_POLICY, TRAJECTORY_LATE_CHECKPOINT_SCHEMA, TRAJECTORY_LATE_SCHEMA, content_sha256, file_record, full_gdr_stage_plan, immutable_json, normalized_trajectory_weights as _normalized_trajectory_weights, selector_lineage, tensor_state_sha256, trajectory_contract_fields, verify_selector_lineage_artifacts

CONFIG_SCHEMA = TRAJECTORY_LATE_SCHEMA

CHECKPOINT_SCHEMA = TRAJECTORY_LATE_CHECKPOINT_SCHEMA

LEGACY_CHECKPOINT_SCHEMA = 'blossom-deltarec-full-gdr-trajectory-v1'

STOP_REQUESTED = False

def read(path):
    return json.loads(Path(path).read_text())

def normalized_trajectory_weights(count, explicit=None):
    """Return positive normalized teacher probabilities for arbitrary T."""

    return _normalized_trajectory_weights(count, explicit)

def checkpoint_contract(c):
    """Fields that determine whether a dense checkpoint is reusable."""

    # Keep the small legacy fixture contract accepted by the CPU tests and by
    # the read-only v1 loader.  New artifacts always take the complete shared
    # contract, including backbone/variant/protocol identity.
    if all(key in c for key in ('backbone', 'variant', 'protocol_sha256')):
        return trajectory_contract_fields(c)
    return dict(dataset=c['dataset'], seed=c['seed'], backend=c['backend'],
        architecture=c['architecture'], grouping_sha256=c['grouping_sha256'],
        group_count=c['group_count'], objective=c['objective'], negatives=c['negatives'],
        temperature=c['temperature'], learning_rate=c['learning_rate'],
        effective_batch=c['effective_batch'], kernel=c['kernel'])

def _legacy_checkpoint_compatible(observed, legacy_config, expected_config):
    """Compare a v1 dense checkpoint against the explicit v2 contract.

    The old Blossom trajectory artifact predates the shared contract fields,
    but its payload has a complete backbone/optimizer/objective contract.  It
    is therefore safe to use as a read-only prefix only after mapping those
    fields explicitly.  In particular, a legacy selector is never accepted
    by this helper.
    """

    if observed != checkpoint_contract(legacy_config):
        return False
    expected = checkpoint_contract(expected_config)
    mapped = dict(
        dataset=legacy_config.get('dataset'), backbone='blossom', seed=legacy_config.get('seed'),
        variant='blossomrec-gdr', architecture=legacy_config.get('architecture'),
        grouping_sha256=legacy_config.get('grouping_sha256'),
        group_count=legacy_config.get('group_count'), objective=legacy_config.get('objective'),
        negatives=legacy_config.get('negatives'), temperature=legacy_config.get('temperature'),
        learning_rate=legacy_config.get('learning_rate'),
        effective_batch=legacy_config.get('effective_batch'), kernel=legacy_config.get('kernel'),
        # v1 did not put the protocol digest into the checkpoint contract.  It
        # is checked against the old workspace binding below.
        protocol_sha256=expected.get('protocol_sha256'),
    )
    if any(mapped.get(key) != value for key, value in expected.items()
           if key != 'protocol_sha256'):
        return False
    binding_path = legacy_config.get('binding')
    binding_hash = legacy_config.get('binding_sha256')
    if not binding_path or not binding_hash:
        return False
    try:
        if sha256_file(binding_path) != binding_hash:
            return False
        binding = read(binding_path)
    except (OSError, ValueError, TypeError):
        return False
    return binding.get('protocol_sha256') == expected.get('protocol_sha256')

def expected_trajectory_paths(c):
    reused = c.get('trajectory_checkpoint_paths') or []
    count = int(c.get('trajectory_count', c.get('full_gdr_warmup_epochs', 0)))
    if count < len(reused):
        raise ValueError('trajectory_count is smaller than reused checkpoints')
    root = Path(c['output']) / 'full_gdr_trajectory'
    return [Path(path) for path in reused] + [
        root / f'epoch-{epoch:03d}.pt'
        for epoch in range(len(reused) + 1, count + 1)
    ]

def _read_trajectory_checkpoint(path, c):
    """Validate/read a trajectory checkpoint without mutating a live model."""

    payload = torch.load(path, map_location='cpu', weights_only=False)
    schema = payload.get('schema')
    legacy_compatible = False
    if schema == CHECKPOINT_SCHEMA:
        observed = payload.get('contract')
        if payload.get('checkpoint_content_sha256'):
            actual = tensor_state_sha256(payload.get('model', {}))
            if actual != payload['checkpoint_content_sha256']:
                raise ValueError(f'trajectory checkpoint content hash changed: {path}')
    elif schema in {LEGACY_CHECKPOINT_SCHEMA, 'blossom-deltarec-full-gdr-best-v1'}:
        observed = checkpoint_contract(payload['config'])
        legacy_compatible = True
    else:
        raise ValueError(f'{path} is not a Blossom Full-GDR checkpoint: {schema!r}')
    expected = checkpoint_contract(c)
    if observed != expected:
        if not legacy_compatible or not _legacy_checkpoint_compatible(
            observed, payload.get('config', {}), c
        ):
            comparable = set(expected) & set(observed or {})
            changed = sorted(key for key in comparable if observed.get(key) != expected.get(key))
            changed = sorted(set(changed) | (set(expected) - set(observed or {})))
            raise ValueError(f'incompatible Full-GDR checkpoint {path}; changed fields: {changed}')
        payload['_trajectory_compatibility'] = 'legacy-dense-checkpoint-read-only'
    return payload

def load_trajectory_checkpoint(path, c, model, *, backend=legacy):
    """Load one compatible Blossom Full-GDR checkpoint without rewriting it."""

    payload = _read_trajectory_checkpoint(path, c)
    backend.load_base_model_state(model, payload['model'])
    return payload

def _write_trajectory_manifest(c, paths):
    """Publish ordered checkpoint lineage separately from immutable tensors."""

    total_count = int(
        c.get('trajectory_count')
        or c.get('full_gdr_warmup_epochs')
        or len(paths)
    )
    weights = normalized_trajectory_weights(total_count, c.get('trajectory_weights'))
    if len(paths) > total_count:
        raise ValueError('trajectory manifest contains more checkpoints than its contract')
    records = []
    previous_epoch = 0
    previous_step = -1
    for index, path in enumerate(paths, 1):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload.get('model', {})
        epoch = int(payload.get('epoch', index))
        raw_step = payload.get('global_step')
        step = int(raw_step if raw_step is not None else payload.get('step', 0))
        if epoch <= previous_epoch or step < previous_step:
            raise ValueError('Blossom trajectory checkpoint order is not increasing')
        if payload.get('schema') == CHECKPOINT_SCHEMA:
            if payload.get('trajectory_index') != index:
                raise ValueError('native Blossom trajectory index is not contiguous')
            if index > 1 and payload.get('parent_checkpoint') != str(Path(paths[index - 2]).resolve()):
                raise ValueError('native Blossom trajectory parent is not the previous checkpoint')
        parent_checkpoint = payload.get('parent_checkpoint')
        if parent_checkpoint is None and index > 1:
            # Legacy v1 payloads did not carry lineage fields.  The new
            # manifest records their immutable ordered-prefix relationship
            # without modifying the legacy checkpoint itself.
            parent_checkpoint = str(Path(paths[index - 2]).resolve())
        parent_checkpoint_sha256 = payload.get('parent_checkpoint_sha256')
        if parent_checkpoint_sha256 is None and parent_checkpoint is not None:
            parent_checkpoint_sha256 = sha256_file(parent_checkpoint)
        records.append(dict(
            trajectory_index=index, path=str(Path(path).resolve()), sha256=sha256_file(path),
            epoch=epoch, global_step=step,
            parent_checkpoint=parent_checkpoint,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            checkpoint_content_sha256=payload.get('checkpoint_content_sha256', tensor_state_sha256(state)),
            teacher_weight=weights[index - 1],
            compatibility=payload.get(
                '_trajectory_compatibility',
                'legacy-dense-checkpoint-read-only'
                if payload.get('schema') in {
                    LEGACY_CHECKPOINT_SCHEMA,
                    'blossom-deltarec-full-gdr-best-v1',
                }
                else 'native-v2',
            ),
        ))
        previous_epoch = epoch
        previous_step = step
    manifest = dict(
        schema='deltarec-blossom-trajectory-lineage-v2', protocol_schema=CONFIG_SCHEMA,
        dataset=c['dataset'], backbone=c['backbone'], variant=c['variant'], seed=c['seed'],
        contract=checkpoint_contract(c), trajectory_count=total_count,
        checkpoints_present=len(paths), trajectory_complete=len(paths) == total_count,
        trajectory_sampling='late',
        target_policy=TARGET_POLICY, normalized_trajectory_weights=weights,
        checkpoints=records, selector_feature_snapshot='final-theta-T-item-table',
        optimizer_scheduler_rng_restored=False,
    )
    manifest['manifest_content_sha256'] = content_sha256(manifest, 'manifest_content_sha256')
    manifest_path = Path(c['output']) / 'trajectory_manifest.json'
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous.get('manifest_content_sha256') != content_sha256(
            previous, 'manifest_content_sha256'
        ):
            raise ValueError('existing Blossom trajectory lineage checksum changed')
        previous_checkpoints = previous.get('checkpoints', [])
        if (
            previous.get('schema') != manifest['schema']
            or previous.get('contract') != manifest['contract']
            or previous.get('trajectory_count') != manifest['trajectory_count']
            or previous.get('normalized_trajectory_weights')
            != manifest['normalized_trajectory_weights']
            or previous.get('target_policy') != manifest['target_policy']
            or not isinstance(previous_checkpoints, list)
            or len(previous_checkpoints) > len(records)
            or previous_checkpoints != records[: len(previous_checkpoints)]
        ):
            raise ValueError('existing Blossom trajectory lineage is not an immutable prefix')
        if len(previous_checkpoints) == len(records):
            immutable_json(manifest_path, manifest)
            return manifest
    atomic_write_json(manifest_path, manifest)
    return manifest

def save_trajectory_checkpoint(path, c, model, *, epoch, step, trajectory_index=None,
                               parent_checkpoint=None, continuation=False,
                               backend=legacy):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        existing_schema = existing.get('schema')
        native = existing_schema == CHECKPOINT_SCHEMA
        legacy = existing_schema in {LEGACY_CHECKPOINT_SCHEMA, 'blossom-deltarec-full-gdr-best-v1'}
        compatible = (
            existing.get('contract') == checkpoint_contract(c)
            if native
            else legacy and _legacy_checkpoint_compatible(
                existing.get('contract'), existing.get('config', {}), c
            )
        )
        if not compatible:
            raise ValueError(f'refusing to overwrite incompatible checkpoint: {path}')
        if native:
            expected_hash = existing.get('checkpoint_content_sha256')
            if expected_hash and expected_hash != tensor_state_sha256(existing.get('model', {})):
                raise ValueError(f'existing trajectory checkpoint checksum changed: {path}')
        return
    state = backend.base_model_state(model)
    index = int(trajectory_index if trajectory_index is not None else epoch)
    count = int(c.get(
        'trajectory_count',
        len(c.get('trajectory_weights') or [])
        or max(index, len(c.get('trajectory_checkpoint_paths', []))),
    ))
    weights = normalized_trajectory_weights(count, c.get('trajectory_weights'))
    if index < 1 or index > count:
        raise ValueError(f'invalid Blossom trajectory index {index} for count {count}')
    parent = None if parent_checkpoint is None else Path(parent_checkpoint).resolve()
    atomic_torch_save(torch, dict(schema=CHECKPOINT_SCHEMA,
        contract=checkpoint_contract(c), config=c, model=state,
        epoch=epoch, global_step=step, step=step, stage='full_gdr_trajectory',
        trajectory_index=index, trajectory_teacher_weight=weights[index - 1],
        parent_checkpoint=None if parent is None else str(parent),
        parent_checkpoint_sha256=None if parent is None else sha256_file(parent),
        continuation_mode='model-only-fresh-optimizer-and-schedule' if continuation else 'scratch-or-true-model-state',
        optimizer_restored=False, scheduler_restored=False, rng_restored=False,
        model_fingerprint=c.get('model_fingerprint'),
        run_fingerprint=c.get('run_fingerprint'),
        protocol_config_sha256=c.get('config_content_sha256'),
        checkpoint_content_sha256=tensor_state_sha256(state)), path)
    manifest_paths = expected_trajectory_paths(c)
    if len(manifest_paths) >= index and manifest_paths[index - 1] == path:
        _write_trajectory_manifest(c, manifest_paths[:index])

def generate_cwi_records(c, model, files, catalog_ids, output, checkpoint_path, index, sampled,
                         *, backend=legacy):
    cwi_root = output / 'cwi_trajectory'
    cwi_root.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(checkpoint_path)
    path = cwi_root / f'teacher-{index:03d}.pt'
    binding = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                   index=index, data_binding_sha256=c['binding_sha256'])
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if (
            payload.get('schema') != 'blossom-deltarec-cwi-trajectory-shard-v2'
            or payload.get('binding') != binding
            or payload.get('target_policy') != TARGET_POLICY
        ):
            raise ValueError(f'CWI shard {path} has a different teacher binding')
        manifest_path = cwi_root / f'teacher-{index:03d}.json'
        if not manifest_path.is_file():
            raise ValueError(f'CWI shard exists without its immutable manifest: {path}')
        manifest = read(manifest_path)
        if manifest.get('target_policy') != TARGET_POLICY:
            raise ValueError(f'CWI shard {path} uses a different target policy')
        return payload['records']
    records = []
    torch.manual_seed(c['seed'] + 2000 + index)
    for batch_index, start in enumerate(range(0, len(sampled), c['cwi_microbatch'])):
        rows = sampled[start:start + c['cwi_microbatch']]
        batch = backend.collate(c, rows, model.item_embedding.weight.device)
        candidates = backend.candidates_for_loss(c, batch, catalog_ids)
        gates = torch.ones(batch['histories'].shape[0], model.group_count,
            batch['histories'].shape[1], device=catalog_ids.device, requires_grad=True)
        with torch.autocast(device_type=model.item_embedding.weight.device.type, enabled=False):
            history = model.prefill(batch['histories'], batch['lengths'], sparse=False,
                                    event_gates=gates)
            loss = backend.recommendation_loss(c, model, history, batch, candidates)
        gradient, = torch.autograd.grad(loss, gates)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('nonfinite teacher CWI labels')
        records.append(dict(history=batch['histories'].cpu(), lengths=batch['lengths'].cpu(),
            users=batch['users'].cpu(), importance=-gradient.detach().cpu(),
            active_groups=F.one_hot(model.item_to_group[candidates], model.group_count).any(1).cpu()))
        if batch_index % 8 == 0:
            backend.emit(output, 'trajectory-cwi-labels', teacher=index,
                        batches=batch_index + 1, total=c['cwi_batches'], loss=float(loss.detach()))
    atomic_torch_save(torch, dict(schema='blossom-deltarec-cwi-trajectory-shard-v2',
        binding=binding, target_policy=TARGET_POLICY, records=records), path,
        overwrite=False)
    manifest = dict(
        schema='deltarec-blossom-cwi-teacher-manifest-v2', dataset=c['dataset'],
        backbone=c.get('backbone', 'blossom'), seed=c['seed'], trajectory_index=index,
        parent_checkpoint=file_record(checkpoint_path),
        data_binding_sha256=c['binding_sha256'], target_policy=TARGET_POLICY,
        shard=file_record(path), rows=len(records), label_space='signed-CWI1-per-group-event',
        validation_candidates_mounted=False, test_data_mounted=False,
    )
    manifest['manifest_content_sha256'] = content_sha256(
        manifest, 'manifest_content_sha256'
    )
    immutable_json(cwi_root / f'teacher-{index:03d}.json', manifest)
    return records

def _selector_record_loss(model, record, device, *, heldout, backend=legacy):
    partition = record['users'].remainder(5).eq(0)
    mask = partition if heldout else ~partition
    if not bool(mask.any()):
        return None
    histories = record['history'][mask].to(device)
    lengths = record['lengths'][mask].to(device)
    importance = record['importance'][mask].to(device)
    active = record['active_groups'][mask].to(device)
    scores = model.selector_scores(histories)
    valid = torch.arange(histories.shape[1], device=device)[None, None] < lengths[:, None, None]
    valid = valid.expand_as(scores)
    loss = backend.cwi_selector_loss(scores[active], importance[active], valid[active])
    return loss, int(active.sum())

def _loss_denominator(c, batch):
    """Return the denominator used by the published backbone objective.

    Rating CE is a per-user mean.  The KuaiRec compatibility path returns the
    weighted sum of eight per-impression BCE means, so its numerator seed is
    the number of impressions (not merely the number of users).  Keeping this
    at the runner seam lets the shared low-sync accumulator preserve each
    backbone's original loss weighting.
    """

    if c['dataset'] == 'kuairand-1k':
        return int(batch['labels'].shape[0] * batch['labels'].shape[1])
    return int(batch['targets'].shape[0])

def fit_trajectory_selector(c, model, files, catalog_ids, output, trajectory_paths,
                            *, backend=legacy):
    """Distill one selector from separately sampled checkpoint observations."""

    device = catalog_ids.device
    sampled = backend.sample_training_rows(c, files, c['cwi_batches'] * c['cwi_microbatch'],
                                           seed=1000, min_history=129)
    if len(trajectory_paths) < 1:
        raise ValueError('trajectory selector requires at least one teacher checkpoint')
    # The final item table is captured once, before any teacher is loaded.  It
    # is the only feature space used by selector training and remains in the
    # adapter's detached snapshot while CWI labels are generated from each
    # teacher's own model state.
    load_trajectory_checkpoint(trajectory_paths[-1], c, model, backend=backend)
    final_table = model.item_embedding.weight.detach().cpu().contiguous().clone()
    final_table_sha256 = tensor_state_sha256({'item_embedding.weight': final_table})
    model.bind_selector_space(catalog_ids, feature_table=final_table)
    datasets = []
    teacher_manifests = []
    for index, path in enumerate(trajectory_paths, 1):
        load_trajectory_checkpoint(path, c, model, backend=backend)
        model.bind_selector_space(catalog_ids, feature_table=final_table)
        model.eval()
        model.requires_grad_(False)
        datasets.append(generate_cwi_records(c, model, files, catalog_ids, output,
                                             path, index, sampled, backend=backend))
        teacher_manifests.append(output / 'cwi_trajectory' / f'teacher-{index:03d}.json')
    weights = normalized_trajectory_weights(len(datasets), c.get('trajectory_weights'))
    cwi_manifest = dict(
        schema='deltarec-blossom-cwi-trajectory-manifest-v2',
        protocol_schema=CONFIG_SCHEMA,
        dataset=c['dataset'], backbone=c.get('backbone', 'blossom'), seed=c['seed'],
        teachers=[read(path) for path in teacher_manifests],
        normalized_trajectory_weights=weights, target_policy=TARGET_POLICY,
        sampling_policy=SAMPLING_POLICY,
        final_theta_T_checkpoint=str(Path(trajectory_paths[-1]).resolve()),
        final_theta_T_checkpoint_sha256=sha256_file(trajectory_paths[-1]),
        final_theta_T_tensor_sha256=final_table_sha256,
        frozen_embedding_snapshot_sha256=final_table_sha256,
        validation_candidates_mounted=False, test_data_mounted=False,
    )
    cwi_manifest['manifest_content_sha256'] = content_sha256(
        cwi_manifest, 'manifest_content_sha256')
    cwi_manifest_path = output / 'cwi_trajectory_manifest.json'
    immutable_json(cwi_manifest_path, cwi_manifest)
    model.selector.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.selector.parameters(),
                                  lr=c['selector_learning_rate'], weight_decay=0.)
    best = float('inf')
    selected = None
    history = []
    sampled_counts = [0] * len(datasets)
    for epoch in range(c['selector_epochs']):
        rng = random.Random(c['seed'] + 3000 + epoch)
        assignments = rng.choices(range(len(datasets)), weights=weights,
                                  k=max(len(records) for records in datasets))
        epoch_counts = [assignments.count(index) for index in range(len(datasets))]
        sampled_counts = [a + b for a, b in zip(sampled_counts, epoch_counts)]
        train_sum = train_count = 0
        # Grouping sampled observations by teacher avoids reloading a potentially
        # multi-GB checkpoint for every mini-batch. Labels remain separate.
        for teacher, records in enumerate(datasets):
            chosen = [position for position, assignment in enumerate(assignments)
                      if assignment == teacher]
            if not chosen:
                continue
            model.selector.train()
            for position in chosen:
                result = _selector_record_loss(model, records[position % len(records)], device,
                                               heldout=False, backend=backend)
                if result is None:
                    continue
                loss, count = result
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite trajectory selector loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.selector.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                train_sum += float(loss.detach()) * count
                train_count += count
        heldout_sum = heldout_count = 0.
        with torch.no_grad():
            for teacher, records in enumerate(datasets):
                model.selector.eval()
                for record in records:
                    result = _selector_record_loss(
                        model, record, device, heldout=True, backend=backend
                    )
                    if result is None:
                        continue
                    loss, count = result
                    heldout_sum += weights[teacher] * float(loss) * count
                    heldout_count += weights[teacher] * count
        if not train_count or not heldout_count:
            raise ValueError('CWI observations do not cover both user partitions')
        row = dict(epoch=epoch, train_loss=train_sum / train_count,
                   heldout_loss=heldout_sum / heldout_count,
                   sampled_teacher_counts=epoch_counts)
        history.append(row)
        if row['heldout_loss'] < best:
            best = row['heldout_loss']
            selected = {key: value.detach().cpu().clone()
                        for key, value in model.selector.state_dict().items()}
        backend.emit(output, 'trajectory-selector-fit', **row)
    model.selector.load_state_dict(selected)
    # Sparse execution always uses theta(T), independently of sampled teachers.
    load_trajectory_checkpoint(trajectory_paths[-1], c, model, backend=backend)
    model.bind_selector_space(catalog_ids, feature_table=final_table)
    model.requires_grad_(True)
    backend.freeze_selector(model)
    atomic_write_json(output / 'selector_fit.json', dict(schema='trajectory-late-v2',
        history=history, best_heldout_loss=best, trajectory_weights_normalized=weights,
        sampled_teacher_counts=sampled_counts,
        target_policy=TARGET_POLICY,
        sampling_policy=SAMPLING_POLICY,
        trajectory_cwi_manifest=str(cwi_manifest_path),
        trajectory_cwi_manifest_sha256=sha256_file(cwi_manifest_path),
        frozen_embedding_snapshot_sha256=final_table_sha256,
        frozen_embedding_snapshot_source='theta_T item embedding table; used for every teacher observation',
        final_selector_embedding_checkpoint=str(trajectory_paths[-1]),
        final_selector_embedding_checkpoint_sha256=sha256_file(trajectory_paths[-1])))

def save_selector(output, c, model, final_path):
    path = output / 'selector.pt'
    state = {key: value.detach().cpu() for key, value in model.selector.state_dict().items()}
    selector_payload = dict(
        schema='deltarec-trajectory-late-selector-checkpoint-v2',
        protocol_schema=CONFIG_SCHEMA,
        config=c,
        state=state,
        selector_state_sha256=tensor_state_sha256(state),
        embedding_ownership='frozen-final-trajectory-snapshot',
        legacy_selector_resume=False,
    )
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        existing_state = existing.get('state', {})
        same_state = (
            isinstance(existing_state, dict)
            and set(existing_state) == set(state)
            and all(torch.equal(existing_state[key], state[key]) for key in state)
        )
        if (
            existing.get('schema') != selector_payload['schema']
            or existing.get('config') != c
            or existing.get('legacy_selector_resume') is not False
            or existing.get('selector_state_sha256')
            != selector_payload['selector_state_sha256']
            or not same_state
        ):
            raise ValueError(f'refusing to overwrite an existing selector: {path}')
    else:
        atomic_torch_save(torch, selector_payload, path)

    trajectory_manifest_path = output / 'trajectory_manifest.json'
    cwi_manifest_path = output / 'cwi_trajectory_manifest.json'
    if not trajectory_manifest_path.is_file() or not cwi_manifest_path.is_file():
        raise ValueError('Trajectory-Late selector provenance manifests are incomplete')
    trajectory_manifest = read(trajectory_manifest_path)
    trajectory = trajectory_manifest.get('checkpoints')
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError('trajectory manifest has no ordered teacher checkpoints')
    cwi_manifest = read(cwi_manifest_path)
    lineage = selector_lineage(
        config=c,
        trajectory=trajectory,
        cwi_manifest=cwi_manifest_path,
        final_snapshot_sha256=cwi_manifest['frozen_embedding_snapshot_sha256'],
        selector_checkpoint=path,
    )
    binding = dict(
        **lineage,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        selector_state_sha256=selector_payload['selector_state_sha256'],
        grouping_sha256=c['grouping_sha256'],
        feature_space='frozen-final-trajectory-item-table',
        feature_checkpoint=str(Path(final_path).resolve()),
        feature_checkpoint_sha256=sha256_file(final_path),
        artifact='trajectory-late-cwi-mlp-no-embedding-rows',
        prototype_reduce='training-catalog-group-means',
        empty_group='training-catalog-global-mean',
    )
    immutable_json(output / 'selector_binding.json', binding)
    return binding

def load_selector(c, model, output, catalog_ids, final_path, *, backend=legacy):
    """Load only a v2 selector and rebind its immutable theta(T) feature space."""

    path = output / 'selector.pt'
    binding_path = output / 'selector_binding.json'
    if not path.is_file() or not binding_path.is_file():
        raise FileNotFoundError('Trajectory-Late selector artifacts are incomplete')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != 'deltarec-trajectory-late-selector-checkpoint-v2':
        raise ValueError('legacy selector checkpoint cannot be used by Trajectory-Late')
    if payload.get('config') != c or payload.get('legacy_selector_resume') is not False:
        raise ValueError('selector checkpoint has incompatible protocol provenance')
    if payload.get('selector_state_sha256') != tensor_state_sha256(payload['state']):
        raise ValueError('selector checkpoint state checksum changed')
    binding = read(binding_path)
    expected_paths = expected_trajectory_paths(c)
    verify_selector_lineage_artifacts(
        binding,
        trajectory_manifest_path=output / 'trajectory_manifest.json',
        final_checkpoint=final_path,
        expected_count=len(expected_paths),
        expected_weights=normalized_trajectory_weights(
            len(expected_paths), c.get('trajectory_weights')
        ),
    )
    if binding.get('path') != str(path.resolve()) or binding.get('sha256') != sha256_file(path):
        raise ValueError('selector binding does not match selector checkpoint')
    if (binding.get('selector_state_sha256') != payload['selector_state_sha256']
            or binding.get('grouping_sha256') != c['grouping_sha256']):
        raise ValueError('selector binding state/grouping provenance changed')
    if binding.get('feature_checkpoint') != str(Path(final_path).resolve()):
        raise ValueError('selector feature snapshot is not the final trajectory checkpoint')
    load_trajectory_checkpoint(final_path, c, model, backend=backend)
    final_table = model.item_embedding.weight.detach().cpu().contiguous().clone()
    model.bind_selector_space(catalog_ids, feature_table=final_table)
    model.selector.load_state_dict(payload['state'], strict=True)
    model.requires_grad_(True)
    backend.freeze_selector(model)
    return binding

def load_best(c, model, output, catalog_ids, *, backend=legacy):
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    if best.get('schema') != 'deltarec-trajectory-late-sparse-best-v2' or best['config'] != c:
        raise ValueError('checkpoint configuration mismatch')
    binding = best['selector_binding']
    final_path = expected_trajectory_paths(c)[-1]
    loaded_binding = load_selector(c, model, output, catalog_ids, final_path,
                                   backend=backend)
    if loaded_binding != binding:
        raise ValueError('sparse checkpoint selector lineage changed')
    backend.load_base_model_state(model, best['model'])
    # The sparse model state is loaded after binding so that its trainable item
    # table is restored while the selector snapshot/prototypes remain theta(T).
    backend.freeze_selector(model)
    return best

def _train_full_gdr(c, model, files, catalog_ids, output, latest, saved, *, backend=legacy):
    paths = expected_trajectory_paths(c)
    reused_count = len(c.get('trajectory_checkpoint_paths') or [])
    completed = 0
    for path in paths:
        if not path.is_file():
            break
        _read_trajectory_checkpoint(path, c)
        completed += 1
    if completed < reused_count:
        raise FileNotFoundError(
            f'reused trajectory has a gap before checkpoint {completed + 1}: {paths[completed]}'
        )
    if completed == len(paths):
        _write_trajectory_manifest(c, paths)
        return paths

    plan = full_gdr_stage_plan(
        saved=saved,
        completed=completed,
        reused_count=reused_count,
        continuation_parent=c.get('continuation_parent_checkpoint'),
    )
    saved_epoch = plan['saved_epoch']
    saved_semantics = plan['saved_semantics']
    resume_local = plan['resume_local']
    # A supplied Full-GDR parent is an intentional model-only continuation.
    # It never inherits Adam, scheduler, or RNG state from the parent run.
    continuation_parent = c.get('continuation_parent_checkpoint')
    continuation = plan['continuation']
    if resume_local:
        optimizer = backend.make_optimizer(c, model)
        if not saved.get('optimizer'):
            raise ValueError('true Full-GDR resume is missing optimizer state')
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch = saved_epoch
        saved_cursor = saved.get('cursor', {})
        global_step = int(saved_cursor.get('global_step', saved_cursor.get('step', 0)))
        resume_semantics = saved_semantics
    elif continuation:
        parent_path = paths[completed - 1] if completed else continuation_parent
        if parent_path is None:
            raise ValueError('continuation has no Full-GDR parent checkpoint')
        parent_path = Path(parent_path).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent_payload = load_trajectory_checkpoint(parent_path, c, model, backend=backend)
        start_epoch = completed
        global_step = int(parent_payload.get('global_step', parent_payload.get('step', 0)))
        optimizer = backend.make_optimizer(c, model)
        resume_semantics = 'continuation-model-only-fresh-optimizer-schedule'
        saved_cursor = {}
    else:
        optimizer = backend.make_optimizer(c, model)
        if saved and saved.get('optimizer'):
            optimizer.load_state_dict(saved['optimizer'])
        start_epoch = saved_epoch
        saved_cursor = saved.get('cursor', {}) if saved else {}
        global_step = int(saved_cursor.get('global_step', saved_cursor.get('step', 0)))
        resume_semantics = 'true-resume-model-optimizer-scheduler-rng'
    accumulation = c['effective_batch'] // c['microbatch']
    controller = FuxiGradientAccumulation(
        accumulation,
        debug_anomaly=bool(c.get('anomaly_detection', False)),
        health_window=int(c.get('startup_health_window', 10)),
        health_interval=int(c.get('health_check_interval', 100)),
        health_path=output / 'full_gdr_step_health.jsonl',
    )
    for epoch in range(start_epoch, c['trajectory_count']):
        model.train()
        backend.freeze_selector(model)
        all_batches = backend.batches(c, files, epoch + 1)
        cursor = saved_cursor if saved and not continuation and epoch == start_epoch else {}
        epoch_loss, examples, step = cursor.get('epoch_loss', 0.), cursor.get('examples', 0), cursor.get('step', 0)
        started = time.time() - cursor.get('elapsed', 0.)
        if step:
            all_batches = itertools.islice(all_batches, step * accumulation, None)
        while window := list(itertools.islice(all_batches, accumulation)):
            denominator = 0
            controller.start_window(model, optimizer, steps=len(window))
            for microbatch_index, rows in enumerate(window):
                controller.start_microbatch(model, microbatch_index)
                batch = backend.collate(c, rows, catalog_ids.device)
                candidates = backend.candidates_for_loss(c, batch, catalog_ids)
                with backend.autocast(model):
                    history = model.prefill(batch['histories'], batch['lengths'],
                                            sparse=False, single_group=True)
                    loss = backend.recommendation_loss(c, model, history, batch, candidates)
                loss_denominator = _loss_denominator(c, batch)
                denominator += loss_denominator
                controller.backward(loss, loss_denominator)
                examples += len(rows)
            health = controller.finish_window(
                model, optimizer, gradient_clip=c['gradient_clip']
            )
            epoch_loss += health['loss'] * denominator
            step += 1
            global_step += 1
            if step % 25 == 0 or step == 1:
                backend.emit(output, 'full-gdr-warmup', epoch=epoch + 1, step=step,
                            global_step=global_step, examples=examples,
                            loss=epoch_loss / examples,
                            detailed_parameter_check=health['detailed_parameter_check'])
            if step % 250 == 0 or STOP_REQUESTED:
                cursor = dict(epoch_loss=epoch_loss, examples=examples, step=step,
                              global_step=global_step,
                              elapsed=time.time() - started)
                backend.save_local(latest, c, model, optimizer, 'full_gdr', epoch=epoch,
                                  cursor=cursor, resume_semantics=resume_semantics)
                if STOP_REQUESTED:
                    backend.emit(output, 'paused-resumable', checkpoint=str(latest),
                                resume_stage='full_gdr', epoch=epoch, step=step,
                                resume_semantics=resume_semantics)
                    return None
        parent = paths[epoch - 1] if epoch else continuation_parent
        save_trajectory_checkpoint(
            paths[epoch], c, model, epoch=epoch + 1, step=global_step,
            trajectory_index=epoch + 1, parent_checkpoint=parent,
            continuation=continuation, backend=backend,
        )
        backend.save_local(latest, c, model, optimizer, 'full_gdr', epoch=epoch + 1,
                          resume_semantics=resume_semantics,
                          cursor=dict(epoch_loss=0., examples=0, step=0,
                                      global_step=global_step, elapsed=0.))
        saved = None
    return paths

def _train_sparse(c, run, model, files, catalog_ids, output, latest, saved, *, backend=legacy):
    backend.freeze_selector(model)
    optimizer = backend.make_optimizer(c, model)
    if saved and saved.get('optimizer'):
        optimizer.load_state_dict(saved['optimizer'])
    early = backend.early_state(c, multitask=run.model.multitask)
    start_epoch = saved.get('epoch', 0) if saved else 0
    if saved and saved.get('early'):
        early = EarlyStoppingState.from_dict(saved['early'])
    accumulation = c['effective_batch'] // c['microbatch']
    controller = FuxiGradientAccumulation(
        accumulation,
        debug_anomaly=bool(c.get('anomaly_detection', False)),
        health_window=int(c.get('startup_health_window', 10)),
        health_interval=int(c.get('health_check_interval', 100)),
        health_path=output / 'sparse_step_health.jsonl',
    )
    for epoch in range(start_epoch, c['max_epochs']):
        model.train()
        backend.freeze_selector(model)
        all_batches = backend.batches(c, files, epoch + 1)
        cursor = saved.get('cursor', {}) if saved and epoch == start_epoch else {}
        epoch_loss, examples, step = cursor.get('epoch_loss', 0.), cursor.get('examples', 0), cursor.get('step', 0)
        loss_units = cursor.get(
            'loss_units',
            examples * (32 if c['dataset'] == 'kuairand-1k' else 1),
        )
        selected, eligible = cursor.get('selected', 0), cursor.get('eligible', 0)
        started = time.time() - cursor.get('elapsed', 0.)
        if step:
            all_batches = itertools.islice(all_batches, step * accumulation, None)
        while window := list(itertools.islice(all_batches, accumulation)):
            denominator = 0
            controller.start_window(model, optimizer, steps=len(window))
            selected_window = torch.zeros((), device=catalog_ids.device, dtype=torch.long)
            eligible_window = torch.zeros((), device=catalog_ids.device, dtype=torch.long)
            for microbatch_index, rows in enumerate(window):
                controller.start_microbatch(model, microbatch_index)
                batch = backend.collate(c, rows, catalog_ids.device)
                candidates = backend.candidates_for_loss(c, batch, catalog_ids)
                with backend.autocast(model):
                    history = model.prefill(batch['histories'], batch['lengths'])
                    loss = backend.recommendation_loss(c, model, history, batch, candidates)
                loss_denominator = _loss_denominator(c, batch)
                denominator += loss_denominator
                controller.backward(loss, loss_denominator)
                examples += len(rows)
                selected_window += history.selected_counts.sum()
                eligible_window += batch['lengths'].sum() * model.group_count
            health = controller.finish_window(
                model, optimizer, gradient_clip=c['gradient_clip']
            )
            epoch_loss += health['loss'] * denominator
            loss_units += denominator
            selected += int(selected_window.item())
            eligible += int(eligible_window.item())
            step += 1
            if step % 25 == 0 or step == 1:
                loss_fields = (
                    {'train_multitask_bce': epoch_loss / max(loss_units, 1)}
                    if c['dataset'] == 'kuairand-1k'
                    else {'loss': epoch_loss / max(loss_units, 1)}
                )
                backend.emit(output, 'sparse-training', epoch=epoch, step=step,
                    examples=examples, realized_write_ratio=selected / max(eligible, 1),
                    **loss_fields)
            if step % 250 == 0 or STOP_REQUESTED:
                cursor = dict(epoch_loss=epoch_loss, loss_units=loss_units,
                              examples=examples, step=step,
                              elapsed=time.time() - started, selected=selected, eligible=eligible)
                backend.save_local(latest, c, model, optimizer, 'sparse', epoch=epoch,
                                  early=early.to_dict(), cursor=cursor)
                if STOP_REQUESTED:
                    backend.emit(output, 'paused-resumable', checkpoint=str(latest),
                                resume_stage='sparse', epoch=epoch, step=step)
                    return None
        result = backend.evaluate(c, model, files, output / 'validation', label=f'epoch-{epoch:03d}')
        metrics = result['metrics']
        if c['dataset'] == 'kuairand-1k':
            backend.emit(
                output,
                'sparse-epoch',
                epoch=epoch,
                train_multitask_bce=epoch_loss / max(loss_units, 1),
                validation_multitask_bce=metrics['multitask_loss'],
                validation_macro_gauc=metrics['macro_gauc'],
                validation_task_gauc=result['details']['task_gauc'],
            )
        decision = early.observe(primary_metric=backend.primary_metric(run, metrics), epoch=epoch,
                                 tie_breaker=metrics.get('multitask_loss'))
        if decision['improved']:
            binding = read(output / 'selector_binding.json')
            atomic_torch_save(torch, dict(schema='deltarec-trajectory-late-sparse-best-v2', config=c,
                model=backend.base_model_state(model), epoch=epoch, validation=result,
                selector_binding=binding,
                parent_trajectory=str(Path(c['output']) / 'trajectory_manifest.json'),
                parent_trajectory_sha256=sha256_file(Path(c['output']) / 'trajectory_manifest.json'),
                sparse_budget=c['retention_ratio'], backbone=c['backbone']), output / 'best.pt')
        backend.save_local(latest, c, model, optimizer, 'sparse', epoch=epoch + 1, early=early.to_dict())
        atomic_write_json(output / 'early_stopping.json', early.to_dict())
        Path(result['evidence']).unlink()
        if decision['should_stop']:
            break
    return early


from deltarec.utils.config import load_trajectory_config

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='DeltaRec BlossomRec trajectory training')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--stop-stage',choices=('full-gdr','selector','sparse'),default='sparse')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args(argv)
    c,run,files=load_trajectory_config(args,'blossomrec')
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    model=legacy.construct(c,run,files)
    catalog_ids=legacy.catalog(files['train_catalog'],args.device)
    if args.evaluate:
        if args.checkpoint is None:parser.error('--checkpoint is required')
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
        print(json.dumps(legacy.evaluate(c,model,files,output/'evaluation',label='evaluation')))
        return
    latest=output/'resume.pt';paths=expected_trajectory_paths(c)
    saved=legacy.restore_local(latest,c,model,catalog_ids,paths[-1]) if latest.exists() else None
    stage=saved['stage'] if saved else 'full_gdr'
    if stage=='full_gdr':
        paths=_train_full_gdr(c,model,files,catalog_ids,output,latest,saved)
        if paths is None:return
        if args.stop_stage=='full-gdr':return
        load_trajectory_checkpoint(paths[-1],c,model)
        stage='selector';saved=None
    if stage=='selector':
        if (output/'selector.pt').exists():load_selector(c,model,output,catalog_ids,paths[-1])
        else:
            fit_trajectory_selector(c,model,files,catalog_ids,output,paths)
            save_selector(output,c,model,paths[-1])
        if args.stop_stage=='selector':return
        legacy.save_local(latest,c,model,legacy.make_optimizer(c,model),'sparse',epoch=0,early=None)
        stage='sparse';saved=None
    if _train_sparse(c,run,model,files,catalog_ids,output,latest,saved) is None:return
    best=load_best(c,model,output,catalog_ids)
    atomic_torch_save(torch,dict(config=json.loads(args.config.read_text()),model=model.state_dict(),validation=best['validation']['metrics']),output/'model.pt')
    print(json.dumps(dict(checkpoint='model.pt',validation=best['validation']['metrics'])))

if __name__=='__main__':
    main()
