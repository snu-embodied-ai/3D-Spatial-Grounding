import shutil
import os
import glob
import pandas as pd

DATA_DIR_ROOT = "/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/synthetic_tabletop"

# for subdir in glob.glob(DATA_DIR_ROOT + '/*'):
#     for sample in glob.glob(subdir + '/*'):
#         new_name = "".join(sample.split("objects/"))

#         shutil.move(sample, new_name)


labels = glob.glob(os.path.join(DATA_DIR_ROOT, "*.csv"))
all_df = [(file.rstrip("_labels.csv")[-1], pd.read_csv(file)) for file in labels]

for num_obj, df in all_df:
    df['id'] = df['id'].apply(lambda x : f"{num_obj}_{x}")

cat_df = pd.concat([i[1] for i in all_df])
cat_df = cat_df.drop(["Unnamed: 0"], axis=1).sample(n=cat_df.shape[0], ignore_index=True)

cat_df.to_csv(os.path.join(DATA_DIR_ROOT, "labels.csv"))