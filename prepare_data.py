import argparse,csv
from pathlib import Path
from deltarec.data.preprocess import preprocess_movielens_20m, preprocess_amazon_books, preprocess_kuairand_1k
from deltarec.data.splits import materialize_rating_splits, materialize_kuai_splits

def write_addressability_catalog(item_map, destination):
    # Address all observed IDs; negative sampling still uses train_catalog.csv.
    with item_map.open(newline="", encoding="utf-8") as source, destination.open("w", newline="", encoding="utf-8") as target:
        reader=csv.DictReader(source);writer=csv.writer(target,lineterminator="\n")
        writer.writerow(["item_id"])
        for row in reader:writer.writerow([int(row["model_id"])])


def main():
    p=argparse.ArgumentParser(description='Original DeltaRec raw-data preprocessing and splits')
    p.add_argument('--dataset',choices=('ml-20m','amazon-books','kuairand-1k'),required=True)
    p.add_argument('--raw',type=Path,nargs='+',required=True)
    p.add_argument('--user-features',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    a=p.parse_args();root=a.data_root/a.dataset
    if a.dataset=='kuairand-1k':
        preprocess_kuairand_1k(a.raw,root,user_features_path=a.user_features)
        materialize_kuai_splits(root/'sequences.csv',root/'splits')
        write_addressability_catalog(root/'item_id_map.csv',root/'splits/full_catalog.csv')
    else:
        if len(a.raw)!=1:p.error('rating datasets require one raw ratings CSV')
        fn=preprocess_movielens_20m if a.dataset=='ml-20m' else preprocess_amazon_books
        fn(a.raw[0],root)
        materialize_rating_splits(root/'sequences.csv',root/'splits',dataset=a.dataset)
    print('Prepared sequences, mapping and splits. Ratings still require the matching frozen candidate asset.')
if __name__=='__main__':main()
