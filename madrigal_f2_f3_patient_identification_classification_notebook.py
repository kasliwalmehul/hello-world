# ## Madrigal F2/F3 Patient Identification Classification Notebook
#
# ### 1. Library Imports/Utils
#
# #### 1.1 Table Loading Utils
#
# In[1]:
spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set(
    "fs.azure.account.oauth.provider.type",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
)
spark.conf.set(
    "fs.azure.account.oauth2.client.id",
    dbutils.secrets.get("databricks-adls", "databricks-adls-nitro-zs-client-id"),
)
spark.conf.set(
    "fs.azure.account.oauth2.client.secret",
    dbutils.secrets.get("databricks-adls", "databricks-adls-nitro-zs-client-secret"),
)
spark.conf.set(
    "fs.azure.account.oauth2.client.endpoint",
    "https://login.microsoftonline.com/547da006-67b6-4d80-b0f3-e5b6a60388dd/oauth2/token",
)

# #### 1.2 Library Imports
#
# In[2]:
import warnings

warnings.filterwarnings("ignore")

# Core Python
import re

# Data manipulation
import pandas as pd
import numpy as np

# Progress bar
from tqdm import tqdm

# Scikit-learn: utilities & model selection
from sklearn.utils import shuffle
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

# Scikit-learn: metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

# Clustering & preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb

# SHAP explainability
import shap

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ### 2. Data Loading
#
# #### 2.1 Load the selected features
#
# In[3]:
# ===============================
# LOAD SHAP IMPORTANCE FROM CSV
# ===============================

import pandas as pd

# Define paths
load_path_shap_train = "/dbfs/FileStore/shap_importance_xgboost_train.csv"
load_path_shap_test = "/dbfs/FileStore/shap_importance_xgboost_test.csv"

# Load Train SHAP
shap_importance_train_loaded = pd.read_csv(load_path_shap_train)
print(f"✅ SHAP Train loaded from: {load_path_shap_train}")
print(f"   Shape: {shap_importance_train_loaded.shape}")

# Load Test SHAP
shap_importance_test_loaded = pd.read_csv(load_path_shap_test)
print(f"✅ SHAP Test loaded from: {load_path_shap_test}")
print(f"   Shape: {shap_importance_test_loaded.shape}")

# Display loaded data
# print("\n" + "=" * 80)
# print("SHAP IMPORTANCE - TRAINING DATA (Top 10)")
# print("=" * 80)
# display(shap_importance_train_loaded.head(5))

# print("\n" + "=" * 80)
# print("SHAP IMPORTANCE - TEST DATA (Top 10)")
# print("=" * 80)
# display(shap_importance_test_loaded.head(5))

# In[4]:
## 193 features selected where normalize importance is >0

selected_features_df = shap_importance_train_loaded[
    shap_importance_train_loaded["normalized_importance"] > 0
]
list_of_features_all = selected_features_df["col_name"].to_list()

len(list_of_features_all)

# In[5]:
# shap_importance_train_loaded

# In[6]:
## select only freq and change in freq features for the model

filtered_df = shap_importance_train_loaded[
    shap_importance_train_loaded["col_name"].str.startswith(
        ("freq_", "change_freq_")
    )
]
# filtered_df

selected_features_df = filtered_df[filtered_df["normalized_importance"] > 0]
list_of_features = selected_features_df["col_name"].to_list()

len(list_of_features)

# In[7]:
# display(shap_importance_train_loaded)

# #### 2.2 Loading MASH Frequency/Recency Features
#
# In[8]:


def read_nitro_parquet_table(
    table_name: str,
    base_path: str = "abfss://raw-veeva-nitro@mdglstlakeprodeus2001.dfs.core.windows.net/nitro/parquet/",
):
    """
    Reads a Nitro parquet table from ADLS Gen2 regardless of partition depth,
    and casts all columns EXCEPT 'patient_id' to Integer.
    """

    path = f"{base_path}{table_name}/"

    try:
        df = spark.read.option("recursiveFileLookup", "true").parquet(path)

        # column to exclude from casting
        exclude_col = "patient_id"

        # build select expressions
        select_exprs = []
        for col in df.columns:
            if col == exclude_col:
                select_exprs.append(F.col(col))
            else:
                select_exprs.append(F.col(col).cast(IntegerType()).alias(col))

        df = df.select(*select_exprs)

        # Dropping duplicates records
        df = df.drop("az_prcsd_ts")
        df = df.dropDuplicates()

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet table '{table_name}' at {path}"
        ) from e


# In[9]:
## For now MASH : remove uncessary features


# MASH
# 1] Recency Features  #########################################################################

parq_mash_recency = read_nitro_parquet_table(
    "p01_patient_identification_MASH_event_selected_recency_backward__c_11_2025_refreshed"
)

# display(parq_mash_recency)
# df_mash_recency = parq_mash_recency.toPandas()
# df_mash_recency.head()


# 2] frequency backward #########################################################################

parq_mash_frequency_backward_part_1 = read_nitro_parquet_table(
    "p01_patient_identification_MASH_event_selected_frequency_backward__c_11_2025_refreshed_part_1"
)
parq_mash_frequency_backward_part_2 = read_nitro_parquet_table(
    "p01_patient_identification_MASH_event_selected_frequency_backward__c_11_2025_refreshed_part_2"
)


# 3] change in frequency backward  #########################################################################

parq_mash_change_frequency_backward_part_1 = read_nitro_parquet_table(
    "p01_patient_identification_MASH_event_selected_Change_frequency_backward__c_11_2025_refreshed_part_1"
)
parq_mash_change_frequency_backward_part_2 = read_nitro_parquet_table(
    "p01_patient_identification_MASH_event_selected_Change_frequency_backward__c_11_2025_refreshed_part_2"
)


# All above needs to merge  #########################################################################

# 1) Join frequency backward parts
mash_frequency_backward = parq_mash_frequency_backward_part_1.join(
    parq_mash_frequency_backward_part_2,
    on="patient_id",
    how="inner",
)

# 2) Join change frequency backward parts
mash_change_frequency_backward = parq_mash_change_frequency_backward_part_1.join(
    parq_mash_change_frequency_backward_part_2,
    on="patient_id",
    how="inner",
)

# 3) Join everything with recency
final_mash_features = (
    parq_mash_recency.join(mash_frequency_backward, on="patient_id", how="left").join(
        mash_change_frequency_backward, on="patient_id", how="left"
    )
)

# sanity check
display(final_mash_features.limit(5))
print("Rows:", final_mash_features.count())
print("Columns:", len(final_mash_features.columns))

# #### 2.3 Loading MASH Frequency/Recency Features
#
# In[10]:
# MASLD || Gold and Unlabelled

# # Recency Features
# reporting_uat_dds__c.p01_patient_identification_MASLD_event_selected_recency_backward__c_11_2025_refreshed

# # frequency backward
# reporting_uat_dds__c.p01_patient_identification_MASLD_event_selected_frequency_backward__c_11_2025_refreshed_part_1
# reporting_uat_dds__c.p01_patient_identification_MASLD_event_selected_frequency_backward__c_11_2025_refreshed_part_2

# # frequency backward
# reporting_uat_dds__c.p01_patient_identification_MASLD_event_selected_Change_frequency_backward__c_11_2025_refreshed_part_1
# reporting_uat_dds__c.p01_patient_identification_MASLD_event_selected_Change_frequency_backward__c_11_2025_refreshed_part_2


# MASLD
# 1] Recency Features  #########################################################################

parq_masld_recency = read_nitro_parquet_table(
    "p01_patient_identification_MASLD_event_selected_recency_backward__c_11_2025_refreshed"
)

# display(parq_masld_recency)
# df_masld_recency = parq_masld_recency.toPandas()
# df_masld_recency.head()


# 2] frequency backward #########################################################################

parq_masld_frequency_backward_part_1 = read_nitro_parquet_table(
    "p01_patient_identification_MASLD_event_selected_frequency_backward__c_11_2025_refreshed_part_1"
)
parq_masld_frequency_backward_part_2 = read_nitro_parquet_table(
    "p01_patient_identification_MASLD_event_selected_frequency_backward__c_11_2025_refreshed_part_2"
)


# 3] change in frequency backward  #########################################################################

parq_masld_change_frequency_backward_part_1 = read_nitro_parquet_table(
    "p01_patient_identification_MASLD_event_selected_Change_frequency_backward__c_11_2025_refreshed_part_1"
)
parq_masld_change_frequency_backward_part_2 = read_nitro_parquet_table(
    "p01_patient_identification_MASLD_event_selected_Change_frequency_backward__c_11_2025_refreshed_part_2"
)


# All above needs to merge  #########################################################################

# 1) Join frequency backward parts
masld_frequency_backward = parq_masld_frequency_backward_part_1.join(
    parq_masld_frequency_backward_part_2,
    on="patient_id",
    how="inner",
)

# 2) Join change frequency backward parts
masld_change_frequency_backward = parq_masld_change_frequency_backward_part_1.join(
    parq_masld_change_frequency_backward_part_2,
    on="patient_id",
    how="inner",
)

# 3) Join everything with recency
final_masld_features = (
    parq_masld_recency.join(masld_frequency_backward, on="patient_id", how="left")
    .join(masld_change_frequency_backward, on="patient_id", how="left")
)

# sanity check
display(final_masld_features.limit(5))
print("Rows:", final_masld_features.count())
print("Columns:", len(final_masld_features.columns))

# #### 2.4 Decide target value 1 or 0
#
# In[11]:


def read_nitro_parquet_table_other_tables(
    table_name: str,
    base_path: str = "abfss://raw-veeva-nitro@mdglstlakeprodeus2001.dfs.core.windows.net/nitro/parquet/",
):
    path = f"{base_path}{table_name}/"

    try:
        df = spark.read.option("recursiveFileLookup", "true").parquet(path)

        # Dropping duplicates records
        df = df.drop("az_prcsd_ts")
        df = df.dropDuplicates()

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet table '{table_name}' at {path}"
        ) from e


# ##### 2.4.1 MASH
#
# In[12]:
from pyspark.sql.functions import col, countDistinct

mash_gold_unlabelled_patients = read_nitro_parquet_table_other_tables(
    "p02_patient_identification_MASH_event_mapped_claims__c_11_2025_refreshed"
)

mash_gold_unlabelled_patients.groupBy("patient_pool_mash").agg(
    countDistinct("patient_id__c").alias("patient_cnt")
).show(truncate=False)

# This is initial count this will change, as we have created features for the events and claims, last 1 year from the anchor dates

# In[13]:
mash_gold_unlabelled_patients_df = (
    mash_gold_unlabelled_patients.select("patient_id__c", "patient_pool_mash")
    .dropDuplicates(["patient_id__c", "patient_pool_mash"])
)
display(mash_gold_unlabelled_patients_df.show(5))

# In[14]:
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_gold_flag_mash(
    df: DataFrame,
    pool_col: str = "patient_pool_mash",
    gold_value: str = "Golden_Pool_Patient",
    output_col: str = "is_gold",
) -> DataFrame:
    """
    Adds a binary column:
      1 if patient is in gold pool
      0 otherwise
    """
    return df.withColumn(
        output_col,
        F.when(F.col(pool_col) == gold_value, F.lit(1)).otherwise(F.lit(0)),
    )


parq_mash_patients_flagged_gold_unlabelled = add_gold_flag_mash(
    mash_gold_unlabelled_patients_df,
    pool_col="patient_pool_mash",
    gold_value="Golden_Pool_Patient",
    output_col="is_gold",
)

display(parq_mash_patients_flagged_gold_unlabelled.show(5))

# In[15]:
# display(final_mash_features.select(*list_of_features))

mash_selected_feature_data = final_mash_features.select(
    *["patient_id"] + list_of_features
)
display(mash_selected_feature_data.show(5))

# In[16]:
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def join_outcome_flag_mash(
    feature_df: DataFrame,
    label_df: DataFrame,
    feature_patient_col: str = "patient_id",
    label_patient_col: str = "patient_id__c",
    outcome_col: str = "is_gold",
    output_outcome_col: str = "outcome_flag",
) -> DataFrame:
    """
    Joins the outcome flag from the label DataFrame to the feature DataFrame for MASH.

    The feature DataFrame has a composite patient_id (e.g., '32162872_2025-01-28'),
    which needs to be split to extract the numeric patient ID for joining.

    Args:
        feature_df: Spark DataFrame containing features with composite patient_id
        label_df: Spark DataFrame containing patient labels (gold/unlabelled)
        feature_patient_col: Column name for patient_id in feature_df (default: 'patient_id')
        label_patient_col: Column name for patient_id in label_df (default: 'patient_id__c')
        outcome_col: Column name for outcome in label_df (default: 'is_gold')
        output_outcome_col: Name for the outcome column in output (default: 'outcome_flag')

    Returns:
        Spark DataFrame with features and outcome_flag column
    """

    # Extract numeric patient_id from composite key (split by '_' and take first element)
    # Also keep original patient_id as it contains anchor date info
    feature_df_with_extracted_id = feature_df.withColumn(
        "extracted_patient_id",
        F.split(F.col(feature_patient_col), "_").getItem(0),
    )

    # Prepare label DataFrame - select only needed columns and rename for clarity
    # Cast patient_id to string for consistent join
    label_df_prepared = (
        label_df.select(
            F.col(label_patient_col).cast("string").alias("label_patient_id"),
            F.col(outcome_col).alias(output_outcome_col),
        ).dropDuplicates(["label_patient_id"])
    )

    # Join on extracted patient_id
    result_df = feature_df_with_extracted_id.join(
        label_df_prepared,
        feature_df_with_extracted_id["extracted_patient_id"]
        == label_df_prepared["label_patient_id"],
        how="inner",
    ).drop("extracted_patient_id", "label_patient_id")

    return result_df


# Join outcome flag to feature data
mash_selected_feature_data_with_outcome = join_outcome_flag_mash(
    feature_df=mash_selected_feature_data,
    label_df=parq_mash_patients_flagged_gold_unlabelled,
)

# Verify the result
display(mash_selected_feature_data_with_outcome.limit(10))
print("Rows:", mash_selected_feature_data_with_outcome.count())
print("Columns:", len(mash_selected_feature_data_with_outcome.columns))

# Check outcome distribution
mash_selected_feature_data_with_outcome.groupBy("outcome_flag").count().show()

# ##### 2.4.2 MASLD
#
# In[17]:
# display(final_masld_features)

# select the features form the list
masld_selected_feature_data = final_masld_features.select(
    *["patient_id"] + list_of_features
)

# display(masld_selected_feature_data)

# In[18]:
# 119 This golden patient having one distinct date || maybe for other dates there might not be the last 1 year data. for now removing this patients
patient_list = [
    49668907,
    5904644,
    28777310,
    46791367,
    34941092,
    56421192,
    33696099,
    14120765,
    2622102,
    5972468,
    37455745,
    6515960,
    4539373,
    11013946,
    41713852,
    13207287,
    1428790,
    3918072,
    39236299,
    34813753,
    44634346,
    5865155,
    48829861,
    33623392,
    6883845,
    6558411,
    26446402,
    43685159,
    16866597,
    14513094,
    45029066,
    7348918,
    18653900,
    17441138,
    40820431,
    34668085,
    18091826,
    30982605,
    6053035,
    25222966,
    26327656,
    24167235,
    45002578,
    33141873,
    12724478,
    13783860,
    47805120,
    44972267,
    8621426,
    10166828,
    26312979,
    40296484,
    45124457,
    48407974,
    26045492,
    27005432,
    31207708,
    41582345,
    25480181,
    43956635,
    14463011,
    44779786,
    54434963,
    46438858,
    39285514,
    27241138,
    38389919,
    47479454,
    36534504,
    21329611,
    9376548,
    2949248,
    9791677,
    35724782,
    4205375,
    10443547,
    14989312,
    6066704,
    14143363,
    13053184,
    43069314,
    53593496,
    18544781,
    22578012,
    36582469,
    22198354,
    45876858,
    31960887,
    15114053,
    31419914,
    47063194,
    49784794,
    745748,
    41488912,
    36850726,
    1723633,
    4270885,
    14451859,
    11899649,
    39778513,
    26199120,
    31446400,
    15719512,
    29285496,
    26667852,
    53190446,
    37511643,
    14338285,
    4659424,
    6257509,
    12787307,
    53085749,
    30918602,
    26795146,
    27411272,
    59805188,
    37021137,
    50111458,
    25656428,
]

# In[19]:


def prepare_patient_anchor_outcome(
    df: DataFrame,
    patient_id_col: str = "patient_id",
    drop_patients: list = None,  # pass list of patient ids to remove (base ids)
    delimiter: str = "_",
    patient_id_out: str = "patient_id_separate",
    anchor_date_out: str = "anchor_date_separate",
    outcome_col: str = "outcome_flag",
) -> DataFrame:
    """
    1) Split patient_id into patient_id_separate and anchor_date_separate
    2) outcome_flag = 1 if patient has >1 distinct anchor_date, else 0
    3) remove patients passed in drop_patients (base patient ids)
    """

    drop_patients = drop_patients or []

    # split columns
    df2 = df.withColumn(
        patient_id_out, F.split(F.col(patient_id_col), delimiter).getItem(0)
    ).withColumn(
        anchor_date_out, F.to_date(F.split(F.col(patient_id_col), delimiter).getItem(1))
    )

    # count distinct anchor dates per patient
    anchor_ct = df2.groupBy(patient_id_out).agg(
        F.countDistinct(anchor_date_out).alias("anchor_date_cnt")
    )

    # join back + create outcome flag
    df3 = (
        df2.join(anchor_ct, on=patient_id_out, how="left")
        .withColumn(
            outcome_col,
            F.when(F.col("anchor_date_cnt") > 1, F.lit(1)).otherwise(F.lit(0)),
        )
        .drop("anchor_date_cnt")
    )

    # filter out explicit patients (base ids)
    if len(drop_patients) > 0:
        df3 = df3.filter(~F.col(patient_id_out).isin([str(x) for x in drop_patients]))

    df3 = df3.drop("anchor_date_separate", "patient_id_separate")
    return df3


parq_masld_gold_unlabelled_with_outcome_data = prepare_patient_anchor_outcome(
    df=masld_selected_feature_data,
    patient_id_col="patient_id",
    drop_patients=patient_list,
)

# display(parq_masld_gold_unlabelled_with_outcome_data)

# ### 3. Data preparation
#
# #### 3.1 Union of MASH and MASHLD
#
# In[20]:
mash_gold_unlabelled_selected_feat_with_outcome_df = (
    mash_selected_feature_data_with_outcome.toPandas()
)

# In[21]:
mash_gold_unlabelled_selected_feat_with_outcome_df.shape

# In[22]:
display(parq_masld_gold_unlabelled_with_outcome_data.limit(10))

# In[23]:


# In[24]:
parq_masld_gold_with_outcome_data = parq_masld_gold_unlabelled_with_outcome_data.filter(
    col("outcome_flag") == 1
)


# In[25]:
masld_gold_selected_feat_with_outcome_df = (
    parq_masld_gold_with_outcome_data.toPandas()
)

# In[26]:
masld_gold_selected_feat_with_outcome_df.shape

# In[27]:
# Classification: MASH/MASLD : ALL Gold Mash/masld ->1 ||| MASH unlabelled --> 0
# select the MASH/MASLD ALL gold patients and MASH unlabelled patients only

# taking masld gold only

# masld_gold_selected_feat_with_outcome_df=masld_gold_unlabelled_selected_feat_with_outcome_df[masld_gold_unlabelled_selected_feat_with_outcome_df['outcome_flag']==1]

# In[28]:
# Union  || MASH = Gold & Unlabelled  || MASLD = Gold only

# Union both DataFrames (concatenate rows where columns match)
combined_df = pd.concat(
    [
        mash_gold_unlabelled_selected_feat_with_outcome_df,
        masld_gold_selected_feat_with_outcome_df,
    ],
    axis=0,
    ignore_index=True,
)

# Verify the result
print(f"MASH rows: {len(mash_gold_unlabelled_selected_feat_with_outcome_df)}")
print(f"MASLD rows: {len(masld_gold_selected_feat_with_outcome_df)}")
print(f"Combined rows: {len(combined_df)}")
print(f"Columns: {len(combined_df.columns)}")

# Check outcome distribution
print("\nOutcome distribution:")
print(combined_df["outcome_flag"].value_counts())

# In[29]:
combined_df.shape

# #### 3.2 Imputation
#
# In[30]:
model_df = combined_df.copy()

# In[31]:
big_number = 9999

model_df.columns = model_df.columns.str.lower()

# Impute columns ending with "_event_time_min" with a big number
model_df.loc[:, model_df.columns.str.contains("_event_time_min")] = model_df.loc[
    :, model_df.columns.str.contains("_event_time_min")
].fillna(big_number)

# Impute other columns with 0
model_df.fillna(0, inplace=True)

# In[32]:
# =====================
# SAVE TO /dbfs/ PATH
# =====================

# Define DBFS path (use /dbfs/ prefix for pandas)
output_path_model_df = "/dbfs/FileStore/model_df_classification_xgboost_only_freq_change_freq_patients_lvl_split.csv"

# Save
model_df.to_csv(output_path_model_df, index=False)
# print(f"✅  Train saved to: {output_path_model_df}")
# print(f"   Shape: {model_df.shape}")


# # Start Run from Here

# In[33]:
import pandas as pd

# Define paths
output_path_model_df = "/dbfs/FileStore/model_df_classification_xgboost_only_freq_change_freq_patients_lvl_split.csv"

model_df = pd.read_csv(output_path_model_df)

# In[34]:
model_df.head()

# In[35]:
# MASH: gold unlabelled data.
# display(mash_gold_unlabelled_patients_df.limit(10))

display(parq_mash_patients_flagged_gold_unlabelled.limit(10))

# In[36]:
mash_gold_unlabelled_flag_table = parq_mash_patients_flagged_gold_unlabelled.toPandas()
mash_gold_unlabelled_flag_table.head()

# In[37]:
# MASLD: Gold
from pyspark.sql.functions import col, split, max as spark_max, when

patient_pool_df = (
    parq_masld_gold_unlabelled_with_outcome_data.withColumn(
        "pat_id", split(col("patient_id"), "_").getItem(0)
    )
    .groupBy("pat_id")
    .agg(spark_max("outcome_flag").alias("is_gold"))
    .withColumn(
        "patient_pool",
        when(col("is_gold") == 1, "Golden_Pool_Patient").otherwise(
            "Unlabelled_Pool_Patients"
        ),
    )
    .select("pat_id", "patient_pool", "is_gold")
)

display(patient_pool_df.limit(10))

# In[38]:

masld_gold_patients_df = patient_pool_df.filter(col("is_gold") == 1)
display(masld_gold_patients_df.limit(10))

# In[39]:
masld_gold_flag_table = masld_gold_patients_df.toPandas()
masld_gold_flag_table.head()

# In[40]:
mash_gold_unlabelled_flag_table.columns

# In[41]:
masld_gold_flag_table.columns

# In[42]:
mash_gold_unlabelled_flag_table["Indication"] = "MASH"
masld_gold_flag_table["Indication"] = "MASLD"

# In[43]:
mash_gold_unlabelled_flag_table.rename(
    columns={
        "patient_id__c": "pat_id",
        "patient_pool_mash": "patient_pool_mash_masld",
    },
    inplace=True,
)

masld_gold_flag_table.rename(
    columns={"patient_pool": "patient_pool_mash_masld"}, inplace=True
)

# In[44]:
mash_gold_unlabelled_flag_table.columns

# In[45]:
masld_gold_flag_table.columns

# In[46]:
mash_masld_flag_table_combine = pd.concat(
    [mash_gold_unlabelled_flag_table, masld_gold_flag_table], ignore_index=True
)

# In[47]:
mash_gold_unlabelled_flag_table.shape, masld_gold_flag_table.shape, mash_masld_flag_table_combine.shape

# In[48]:
mash_masld_flag_table_combine.head()

# In[49]:
941843 + 15336

# In[50]:
# save mash masld combine table
# =====================
# SAVE TO /dbfs/ PATH
# =====================

# Define DBFS path (use /dbfs/ prefix for pandas)
output_path_flag = (
    "/dbfs/FileStore/mash_masld_flag_table_only_freq_change_freq_patients_lvl_split.csv"
)

# Save
mash_masld_flag_table_combine.to_csv(output_path_flag, index=False)

# ## load mash/masld combine table direclty
# -- mash- gold unlabelled
# -- masld- gold
#

# In[51]:
import pandas as pd

# Define paths
output_path_flag = (
    "/dbfs/FileStore/mash_masld_flag_table_only_freq_change_freq_patients_lvl_split.csv"
)

mash_masld_flag_table_combine = pd.read_csv(output_path_flag)

# In[52]:
mash_masld_flag_table_combine.head()

#

# ##### 3.2.1 Find the patients anchor in last 2 months from Oct 2025
#
# In[53]:
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_gold_last_n_months(
    df,
    patient_id_col="patient_id",
    outcome_col="outcome_flag",
    reference_date="2025-10-31",
    months_back=2,
):
    """
    Get Gold patients (outcome_flag=1) with anchor date in last N months.
    """

    # Convert reference date
    ref_date = pd.to_datetime(reference_date)
    cutoff_date = ref_date - relativedelta(months=months_back)

    df = df.copy()

    # Debug: Check patient_id format
    print("Sample patient_id values:")
    print(df[patient_id_col].head(3).tolist())

    # Extract anchor date - split by '_' and take everything after first '_'
    # Format: 23709322_2025-09-30 → take '2025-09-30'
    df["_anchor_date"] = pd.to_datetime(
        df[patient_id_col].str.split("_", n=1).str[1],
        format="%Y-%m-%d",
        errors="coerce",
    )

    print(f"\nmin_date: {df['_anchor_date'].min()}")
    print(f"max_date: {df['_anchor_date'].max()}")

    # Filter: Gold patients only + last N months
    mask = (df[outcome_col] == 1) & (df["_anchor_date"] >= cutoff_date) & (
        df["_anchor_date"] <= ref_date
    )

    df_gold_last_n_months = df[mask].drop(columns=["_anchor_date"]).reset_index(
        drop=True
    )

    # Print summary
    print(f"\n{'='*55}")
    print(f"  GOLD PATIENTS - LAST {months_back} MONTHS")
    print(f"{'='*55}")
    print(f"  Date Range:        {cutoff_date.date()} to {ref_date.date()}")
    print(f"  Gold Patients:     {len(df_gold_last_n_months):,}")
    print(f"{'='*55}")

    return df_gold_last_n_months


# =============================================================================
# Usage
# =============================================================================

df_gold_last_2_months = get_gold_last_n_months(
    df=model_df,
    patient_id_col="patient_id",
    outcome_col="outcome_flag",
    reference_date="2025-10-31",
    months_back=2,
)

# In[54]:

df_gold_last_2_months.shape

# In[55]:
df_gold_last_2_months.head()

# In[56]:
## Just checking

dates = pd.to_datetime(df_gold_last_2_months["patient_id"].str.rsplit("_", n=1).str[1])

print(f"Min anchor date: {dates.min()}")
print(f"Max anchor date: {dates.max()}")

# In[57]:

last_2months_gold_patient_id = (
    df_gold_last_2_months["patient_id"].str.rsplit("_", n=1).str[0].unique()
)

print(f"Total unique patients: {len(last_2months_gold_patient_id)}")
last_2months_gold_patient_id

# #### Use this flag table for further operation like we have removed the last 2 months patients from here
#
#

# In[58]:
mash_masld_flag_table_combine["pat_id"] = mash_masld_flag_table_combine[
    "pat_id"
].astype(str)

# In[59]:
mash_masld_flag_table_combine_removed_last_2month = mash_masld_flag_table_combine[
    ~mash_masld_flag_table_combine["pat_id"].isin(last_2months_gold_patient_id)
]

print(f"Before: {len(mash_masld_flag_table_combine)}")
print(f"After: {len(mash_masld_flag_table_combine_removed_last_2month)}")
print(
    f"Removed: {len(mash_masld_flag_table_combine) - len(mash_masld_flag_table_combine_removed_last_2month)}"
)

# In[60]:
mash_masld_flag_table_combine[mash_masld_flag_table_combine["pat_id"] == "6967162"]

# In[61]:
mash_masld_flag_table_combine_removed_last_2month.head()

# In[62]:
# Patients level pe split the Data.

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    mash_masld_flag_table_combine_removed_last_2month,
    test_size=0.2,
    random_state=42,
    stratify=mash_masld_flag_table_combine_removed_last_2month["is_gold"],
)

print(f"Total: {len(mash_masld_flag_table_combine_removed_last_2month)}")
print(f"Train: {len(train_df)}")
print(f"Test: {len(test_df)}")

# Verify stratification
print("\nTrain is_gold distribution:")
print(train_df["is_gold"].value_counts(normalize=True))

print("\nTest is_gold distribution:")
print(test_df["is_gold"].value_counts(normalize=True))

# In[63]:
model_df.head(2)

# In[64]:
train_df.head(2)

# In[65]:
test_df.head(2)

# In[66]:
# Get train and test patient IDs
train_patient_ids = train_df["pat_id"].unique().tolist()
test_patient_ids = test_df["pat_id"].unique().tolist()

# Extract patient_id (without anchor date) from model_df
model_df["pat_id"] = model_df["patient_id"].str.rsplit("_", n=1).str[0]

# Filter model_df based on train and test patient IDs
train_data = model_df[model_df["pat_id"].isin(train_patient_ids)].copy()
test_data = model_df[model_df["pat_id"].isin(test_patient_ids)].copy()

# Separate features and target
# Assuming 'outcome_flag' is your target column, adjust if different
feature_cols = [
    col for col in model_df.columns if col not in ["patient_id", "pat_id", "outcome_flag"]
]

X_train = train_data[feature_cols]
y_train = train_data["outcome_flag"]

X_test = test_data[feature_cols]
y_test = test_data["outcome_flag"]

# Set patient_id as index
X_train.index = train_data["patient_id"]
X_test.index = test_data["patient_id"]
y_train.index = train_data["patient_id"]
y_test.index = test_data["patient_id"]

# Summary
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

print(f"\nTrain unique patients: {train_data['pat_id'].nunique()}")
print(f"Test unique patients: {test_data['pat_id'].nunique()}")

print(f"\ny_train distribution:")
print(y_train.value_counts())

print(f"\ny_test distribution:")
print(y_test.value_counts())

# In[67]:
708881 + 172532

# In[68]:
177227 + 43866

# In[69]:
881413 + 221093

# In[70]:


# In[71]:


# ### 4. Modelling
#
# #### 4.1 Prepare Features and Target
#
# ##### 4.1.1 Filter X and Y for the last 2 months GOLD  patients anchor
#
# In[72]:
# Define target and ID columns
target_col = "outcome_flag"
id_col = "patient_id"

df_gold_last_2_months.set_index("patient_id", inplace=True)

# Separate features and target
feature_cols = [
    col for col in df_gold_last_2_months.columns if col not in [target_col, id_col]
]

X_gold_last_2month = df_gold_last_2_months[feature_cols]
y_gold_last_2month = df_gold_last_2_months[target_col]

print(f"Total samples: {len(df_gold_last_2_months)}")
print(f"Number of features: {len(feature_cols)}")
print(f"Target distribution:\n{y_gold_last_2month.value_counts()}")
print(
    f"Class ratio (1s/0s): {y_gold_last_2month.sum() / (len(y_gold_last_2month) - y_gold_last_2month.sum()):.4f}"
)

# In[73]:

import pandas as pd


def get_data_summary(X, y, dataset_name):
    """Get summary statistics for a dataset."""

    # Extract patient_id from index
    patient_ids = y.index.str.rsplit("_", n=1).str[0]
    anchor_dates = pd.to_datetime(y.index.str.rsplit("_", n=1).str[1])

    # Unique patients
    unique_patients = patient_ids.nunique()

    # Outcome distribution
    total_records = len(y)
    positive_count = y.sum()
    negative_count = (y == 0).sum()

    # Unique patients by outcome
    patient_outcome_df = pd.DataFrame({"patient_id": patient_ids, "outcome": y.values})
    unique_positive_patients = patient_outcome_df[
        patient_outcome_df["outcome"] == 1
    ]["patient_id"].nunique()
    unique_negative_patients = patient_outcome_df[
        patient_outcome_df["outcome"] == 0
    ]["patient_id"].nunique()

    # Anchor date range
    min_anchor_date = anchor_dates.min()
    max_anchor_date = anchor_dates.max()

    # Features count
    num_features = X.shape[1]

    summary = {
        "Dataset": dataset_name,
        "Total Records (Patient-Anchor)": f"{total_records:,}",
        "Unique Patients": f"{unique_patients:,}",
        "Avg Anchors per Patient": f"{total_records / unique_patients:.2f}",
        "Num Features": f"{num_features:,}",
        "Positive Records (Gold)": f"{positive_count:,}",
        "Negative Records (Unlabelled)": f"{negative_count:,}",
        "Positive Rate": f"{positive_count / total_records * 100:.2f}%",
        "Unique Positive Patients": f"{unique_positive_patients:,}",
        "Unique Negative Patients": f"{unique_negative_patients:,}",
        "Min Anchor Date": min_anchor_date.strftime("%Y-%m-%d"),
        "Max Anchor Date": max_anchor_date.strftime("%Y-%m-%d"),
    }

    return summary


# Get summaries for all datasets
train_summary = get_data_summary(X_train, y_train, "Train")
test_summary = get_data_summary(X_test, y_test, "Test")
gold_2month_summary = get_data_summary(
    X_gold_last_2month, y_gold_last_2month, "Gold Last 2 Months"
)

# Create summary DataFrame
summary_df = pd.DataFrame([train_summary, test_summary, gold_2month_summary])
summary_df = summary_df.set_index("Dataset").T

# Print summary
print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(summary_df.to_string())
print("=" * 80)

# Display as table
summary_df

# In[74]:
display()

# #### 4.2 Split Data into Training and Testing set
#
# In[75]:


# #### 4.3 XGBoost Model Training
#

# In[76]:
# =============================================================================
# 3. XGBoost Model Training
# =============================================================================

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")

best_params = {
    # 'objective': 'binary:logistic',
    # 'eval_metric': 'aucpr',
    # 'booster': 'gbtree',
    # 'learning_rate': 0.01,
    # 'n_estimators': 850,
    # 'max_depth': 5,
    # 'min_child_weight': 12,
    # 'subsample': 0.759,
    # 'colsample_bytree': 0.840,
    # 'scale_pos_weight': 3,
    # 'random_state': 42,
    # 'n_jobs': -1

    "objective": "binary:logistic",
    "eval_metric": "auc",
    "booster": "gbtree",
    "n_estimators": 1000,
    "max_depth": 6,
    "min_child_weight": 1,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,  # RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 1,
    "use_label_encoder": False,
    "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
}

# Train model
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train, verbose=True)

# #### 4.4 Predictions
#
# In[77]:
# =============================================================================
# 4. Predictions
# =============================================================================

# Class predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Probability predictions
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# In[78]:
# y_test.reset_index()['outcome_flag'].sum() #.value_counts()

# In[79]:


def analyze_unique_patients(df, patient_id_col="patient_id", outcome_col="outcome_flag"):
    """
    Analyze unique patients from patient_id (split from composite key).
    Get unique patient count and outcome based on max value per patient.
    """

    df = df.copy()

    # Extract patient_id (split by '_' and take first part)
    df["extracted_patient_id"] = df[patient_id_col].str.split("_").str[0]

    # Total records
    total_records = len(df)

    # Unique patients count
    unique_patients = df["extracted_patient_id"].nunique()

    # Sum of outcome_flag for each unique patient (then take max - if any 1, then 1)
    patient_outcome = (
        df.groupby("extracted_patient_id")[outcome_col].max().reset_index()
    )
    patient_outcome.columns = ["patient_id", "outcome_flag"]

    # Counts
    total_patients = len(patient_outcome)
    outcome_1_count = patient_outcome["outcome_flag"].sum()
    outcome_0_count = total_patients - outcome_1_count

    # Print summary
    print(f"{'='*55}")
    print(f"  UNIQUE PATIENTS ANALYSIS")
    print(f"{'='*55}")
    print(f"  Total Records:           {total_records:,}")
    print(f"  Unique Patients:         {unique_patients:,}")
    print(f"{'='*55}")
    print(f"  OUTCOME FLAG (Based on Max per Patient)")
    print(f"{'='*55}")
    print(f"  Total Patients:          {total_patients:,}")
    print(f"  Outcome = 1 (Gold):      {outcome_1_count:,}")
    print(f"  Outcome = 0 (Unlabelled):{outcome_0_count:,}")
    print(f"{'='*55}")

    return patient_outcome


# =============================================================================
# Usage
# =============================================================================

patient_outcome_df = analyze_unique_patients(
    df=y_test.reset_index(),
    patient_id_col="patient_id",
    outcome_col="outcome_flag",
)

# patient_outcome_df

# #### Top K metrics
#
# In[80]:
import pandas as pd

# Create DataFrame for training data
train_df = pd.DataFrame(
    {
        "patient_id": X_train.index,
        "y_flag": y_train,
        "pred_proba": y_train_proba,
        "train_test_flag": "train",
    }
)

# Create DataFrame for test data
test_df = pd.DataFrame(
    {
        "patient_id": X_test.index,
        "y_flag": y_test.values.ravel(),
        "pred_proba": y_test_proba,
        "train_test_flag": "test",
    }
)

# Concatenate both train and test dataframes
pat_proba_df = pd.concat([train_df, test_df])

# Reset index for the final DataFrame (optional)
pat_proba_df.reset_index(drop=True, inplace=True)

# In[81]:
pat_proba_df.train_test_flag.value_counts()

# In[82]:
# pat_proba_df.train_test_flag.value_counts()

# In[83]:
import pandas as pd
import numpy as np


def compute_topk_metrics(
    df,
    outcome_col="outcome_flag",
    score_col="pred_probability",
    k_list=[0.10, 0.15, 0.20, 0.25, 0.30],
):
    """
    Computes Top-K evaluation metrics:
    - # Positives in Top K
    - Recall@K
    - % Positives in Top K
    - Lift@K
    """

    # Sort by predicted probability (descending)
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    total_population = len(df_sorted)
    total_positives = df_sorted[outcome_col].sum()
    baseline_rate = total_positives / total_population

    results = []

    for k in k_list:
        top_k_size = int(np.ceil(k * total_population))
        top_k_df = df_sorted.iloc[:top_k_size]

        positives_in_top_k = top_k_df[outcome_col].sum()

        recall_at_k = positives_in_top_k / total_positives
        pct_positives_in_top_k = positives_in_top_k / top_k_size
        lift_at_k = pct_positives_in_top_k / baseline_rate

        results.append(
            {
                "Top K% Predicted": f"Top {int(k*100)}%",
                "# Positives in Top K": int(positives_in_top_k),
                "Recall@K": round(recall_at_k * 100, 2),
                "% Positives in Top K": round(pct_positives_in_top_k * 100, 2),
                "Lift@K": round(lift_at_k, 2),
            }
        )

    return pd.DataFrame(results)


# In[84]:

metrics_df = compute_topk_metrics(
    df=pat_proba_df[pat_proba_df["train_test_flag"] == "test"],
    outcome_col="y_flag",
    score_col="pred_proba",
    k_list=list(np.arange(0.1, 1.1, 0.1)),
)

metrics_df.head(10)

# In[85]:
display(metrics_df)

# In[86]:

# metrics_df = compute_topk_metrics(
#     df=pat_prob_df[pat_proba_df['train_test_flag']=='test'],
#     outcome_col="y_flag",
#     score_col="pred_proba",
#     k_list=list(np.arange(0.1, 1.1, 0.1))
# )

# metrics_df.head(10)

# In[87]:
pat_proba_df.head()

# In[88]:
pat_proba_df_test = pat_proba_df[pat_proba_df["train_test_flag"] == "test"]
pat_proba_df_test.head()

# In[89]:
# Split patient_id and anchor_date
pat_proba_df_test[["pat_id", "anchor_date"]] = pat_proba_df_test[
    "patient_id"
].str.rsplit("_", n=1, expand=True)
pat_proba_df_test["anchor_date"] = pd.to_datetime(pat_proba_df_test["anchor_date"])

# Get the row with max anchor date for each patient
patient_max_anchor = pat_proba_df_test.loc[
    pat_proba_df_test.groupby("pat_id")["anchor_date"].idxmax()
][["pat_id", "anchor_date", "pred_proba", "y_flag"]].reset_index(drop=True)

# Rename columns
patient_max_anchor.columns = ["patient_id", "max_anchor_date", "pred_proba", "y_flag"]

patient_max_anchor.head()

# In[90]:

patient_df = patient_max_anchor.copy()
patient_df.rename(
    columns={
        "patient_id": "pat_id",
        "pred_proba": "pred_probability",
        "y_flag": "outcome_flag",
    },
    inplace=True,
)

# In[91]:
patient_df.head()

# In[92]:
import numpy as np
import pandas as pd


def compute_topk_metrics(
    patient_df,
    score_col="pred_probability",
    outcome_col="outcome_flag",
    k_list=list(np.arange(0.1, 1.1, 0.1)),
):
    """
    Computes Top-K / Recall / Lift metrics on patient-level data.
    """

    df = patient_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    total_population = len(df)
    total_positives = df[outcome_col].sum()
    baseline_rate = total_positives / total_population

    results = []

    for k in k_list:
        top_k_size = int(np.ceil(k * total_population))
        top_k_df = df.iloc[:top_k_size]

        positives_in_top_k = top_k_df[outcome_col].sum()

        results.append(
            {
                "Top K% Predicted": f"Top {int(k * 100)}%",
                "# Positives in Top K": int(positives_in_top_k),
                "Recall@K (%)": round(100 * positives_in_top_k / total_positives, 2),
                "% Positives in Top K": round(100 * positives_in_top_k / top_k_size, 2),
                "Lift@K": round((positives_in_top_k / top_k_size) / baseline_rate, 2),
            }
        )

    return pd.DataFrame(results)


# In[93]:

pat_metrics_df = compute_topk_metrics(patient_df)
pat_metrics_df.head(10)

# In[94]:
display(pat_metrics_df)

# In[95]:
y_test.reset_index()["outcome_flag"].sum()

# In[96]:
X_test.shape, y_test.shape

# In[97]:
# patients anchor level baseline rate %

total_test_population = X_test.shape[0]
total_positive = y_test.reset_index()["outcome_flag"].sum()
print("total_populations: ", total_test_population)
print("total_positive: ", total_positive)
print("baseline_rate: ", round(total_positive / total_test_population, 2))

# In[98]:
## patient level baseline rate %

total_test_population_pat_level = patient_df.shape[0]
total_positive_pat_level = patient_df["outcome_flag"].sum()
print("total_populations: ", total_test_population_pat_level)
print("total_positive: ", total_positive_pat_level)
print("baseline_rate: ", round(total_positive_pat_level / total_test_population_pat_level, 2))

# #### 4.5 Evaluation Metrics
#
# In[99]:
y_train.head()

# In[100]:
y_train.shape

# In[101]:
y_train_pred

# In[102]:
len(y_train_pred)

# In[103]:
y_train_proba

# In[104]:
len(y_train_proba)

# In[105]:
# =============================================================================
# 5. Evaluation Metrics || Patient anchor level
# =============================================================================


def print_metrics(y_true, y_pred, y_proba, dataset_type="Train"):
    """Print comprehensive classification metrics."""
    print(f"\n{'='*50}")
    print(f"{dataset_type} Metrics:")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


print_metrics(y_train, y_train_pred, y_train_proba, "Train")
print_metrics(y_test, y_test_pred, y_test_proba, "Test")

# Detailed classification report
print("\n" + "=" * 50)
print("Test Set Classification Report:")
print("=" * 50)
print(
    classification_report(
        y_test, y_test_pred, target_names=["Unlabelled (0)", "Gold (1)"]
    )
)

# In[106]:
# =============================================================================
# Patient Level Metrics - Using Max Anchor Date
# =============================================================================


def get_patient_level_data(y_true, y_proba):
    """Get patient level data using max anchor date's probability."""

    # Create dataframe with patient_id_anchor, outcome and probability
    patient_anchor_df = pd.DataFrame(
        {
            "patient_id_anchor": y_true.index,
            "outcome_flag": y_true.values,
            "probability": y_proba,
        }
    )

    # Split patient_id and anchor_date
    patient_anchor_df[["patient_id", "anchor_date"]] = patient_anchor_df[
        "patient_id_anchor"
    ].str.rsplit("_", n=1, expand=True)
    patient_anchor_df["anchor_date"] = pd.to_datetime(patient_anchor_df["anchor_date"])

    # Get the max anchor date row for each patient
    patient_level_df = patient_anchor_df.loc[
        patient_anchor_df.groupby("patient_id")["anchor_date"].idxmax()
    ].reset_index(drop=True)

    return patient_level_df


def print_patient_level_metrics(y_true, y_proba, dataset_type="Train", threshold=0.5):
    """Print comprehensive classification metrics at patient level."""

    # Get patient level data
    patient_df = get_patient_level_data(y_true, y_proba)

    y_true_patient = patient_df["outcome_flag"]
    y_proba_patient = patient_df["probability"]
    y_pred_patient = (y_proba_patient >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"{dataset_type} Metrics (Patient Level - Max Anchor Date):")
    print(f"{'='*50}")
    print(f"Total Patient-Anchor Pairs: {len(y_true)}")
    print(f"Total Unique Patients: {len(patient_df)}")
    print(f"\nAccuracy:  {accuracy_score(y_true_patient, y_pred_patient):.4f}")
    print(f"Precision: {precision_score(y_true_patient, y_pred_patient):.4f}")
    print(f"Recall:    {recall_score(y_true_patient, y_pred_patient):.4f}")
    print(f"F1 Score:  {f1_score(y_true_patient, y_pred_patient):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true_patient, y_proba_patient):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true_patient, y_pred_patient))

    return patient_df


# Get patient level metrics for Train and Test
print("\n" + "=" * 60)
print("PATIENT LEVEL METRICS (Using Max Anchor Date)")
print("=" * 60)

train_patient_df = print_patient_level_metrics(y_train, y_train_proba, "Train")
test_patient_df = print_patient_level_metrics(y_test, y_test_proba, "Test")

# Detailed classification report at patient level
print("\n" + "=" * 50)
print("Test Set Classification Report (Patient Level):")
print("=" * 50)
y_test_patient = test_patient_df["outcome_flag"]
y_test_patient_pred = (test_patient_df["probability"] >= 0.5).astype(int)
print(
    classification_report(
        y_test_patient, y_test_patient_pred, target_names=["Unlabelled (0)", "Gold (1)"]
    )
)

# In[107]:


# In[108]:
############################################################## MODEL PERFORMANCE
# patients anchor level
# ##################################################################
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


def plot_prc(y_true, y_pred_prob, dataset_type="Train"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    prc_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"{dataset_type} PRC (AUC = {prc_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()


# Train PRC
y_train_pred_prob = model.predict_proba(X_train)[:, 1]
plot_prc(y_train, y_train_pred_prob, "Train")

# Test PRC
y_test_pred_prob = model.predict_proba(X_test)[:, 1]
plot_prc(y_test, y_test_pred_prob, "Test")

plt.show()

from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc(y_true, y_pred_prob, dataset_type="Train"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{dataset_type} ROC (AUC = {auc_score:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()


# Train ROC
plot_roc(y_train, y_train_pred_prob, "Train")

# Test ROC
plot_roc(y_test, y_test_pred_prob, "Test")

plt.show()

from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics_vs_threshold(y_train, y_train_pred_prob, y_test, y_test_pred_prob):
    # Calculate metrics for train data
    precision_train, recall_train, thresholds_train = precision_recall_curve(
        y_train, y_train_pred_prob
    )
    f1_train = 2 * (precision_train * recall_train) / (
        precision_train + recall_train
    )

    # Calculate metrics for test data
    precision_test, recall_test, thresholds_test = precision_recall_curve(
        y_test, y_test_pred_prob
    )
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))

    # Train metrics (dimmed lines)
    plt.plot(
        thresholds_train,
        precision_train[:-1],
        "b--",
        label="Train Precision",
        alpha=0.3,
    )
    plt.plot(
        thresholds_train,
        recall_train[:-1],
        "g--",
        label="Train Recall",
        alpha=0.3,
    )
    plt.plot(
        thresholds_train,
        f1_train[:-1],
        "r--",
        label="Train F1 Score",
        alpha=0.3,
    )

    # Test metrics
    plt.plot(
        thresholds_test,
        precision_test[:-1],
        "b-",
        label="Test Precision",
        alpha=0.9,
    )
    plt.plot(thresholds_test, recall_test[:-1], "g-", label="Test Recall", alpha=0.9)
    plt.plot(thresholds_test, f1_test[:-1], "r-", label="Test F1 Score", alpha=0.9)

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision, Recall, and F1 Score vs. Threshold")
    plt.legend(loc="best")
    plt.grid(False)
    plt.show()


# Call the function with your train and test predictions
plot_metrics_vs_threshold(y_train, y_train_pred_prob, y_test, y_test_pred_prob)

# In[109]:
##############################################################
# MODEL PERFORMANCE - PATIENT LEVEL (Using Max Anchor Date)
##################################################################

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def get_patient_level_data(y_true, y_proba):
    """Get patient level data using max anchor date's probability."""

    patient_anchor_df = pd.DataFrame(
        {
            "patient_id_anchor": y_true.index,
            "outcome_flag": y_true.values,
            "probability": y_proba,
        }
    )

    patient_anchor_df[["patient_id", "anchor_date"]] = patient_anchor_df[
        "patient_id_anchor"
    ].str.rsplit("_", n=1, expand=True)
    patient_anchor_df["anchor_date"] = pd.to_datetime(patient_anchor_df["anchor_date"])

    patient_level_df = patient_anchor_df.loc[
        patient_anchor_df.groupby("patient_id")["anchor_date"].idxmax()
    ].reset_index(drop=True)

    return patient_level_df["outcome_flag"].values, patient_level_df["probability"].values


# Get patient level data for train and test
y_train_patient, y_train_patient_proba = get_patient_level_data(
    y_train, y_train_pred_prob
)
y_test_patient, y_test_patient_proba = get_patient_level_data(
    y_test, y_test_pred_prob
)

print(
    f"Train - Patient Anchor Pairs: {len(y_train)}, Unique Patients: {len(y_train_patient)}"
)
print(
    f"Test - Patient Anchor Pairs: {len(y_test)}, Unique Patients: {len(y_test_patient)}"
)

# =============================================================================
# Precision-Recall Curve - Patient Level
# =============================================================================


def plot_prc_patient(y_true, y_pred_prob, dataset_type="Train"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    prc_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"{dataset_type} PRC (AUC = {prc_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Patient Level)")
    plt.legend()


plt.figure(figsize=(8, 6))
plot_prc_patient(y_train_patient, y_train_patient_proba, "Train")
plot_prc_patient(y_test_patient, y_test_patient_proba, "Test")
plt.show()

# =============================================================================
# ROC Curve - Patient Level
# =============================================================================


def plot_roc_patient(y_true, y_pred_prob, dataset_type="Train"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{dataset_type} ROC (AUC = {auc_score:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Patient Level)")
    plt.legend()


plt.figure(figsize=(8, 6))
plot_roc_patient(y_train_patient, y_train_patient_proba, "Train")
plot_roc_patient(y_test_patient, y_test_patient_proba, "Test")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.show()

# =============================================================================
# Precision, Recall, F1 vs Threshold - Patient Level
# =============================================================================


def plot_metrics_vs_threshold_patient(
    y_train, y_train_pred_prob, y_test, y_test_pred_prob
):
    # Calculate metrics for train data
    precision_train, recall_train, thresholds_train = precision_recall_curve(
        y_train, y_train_pred_prob
    )
    f1_train = 2 * (precision_train * recall_train) / (
        precision_train + recall_train + 1e-10
    )

    # Calculate metrics for test data
    precision_test, recall_test, thresholds_test = precision_recall_curve(
        y_test, y_test_pred_prob
    )
    f1_test = 2 * (precision_test * recall_test) / (
        precision_test + recall_test + 1e-10
    )

    plt.figure(figsize=(10, 6))

    # Train metrics (dimmed lines)
    plt.plot(
        thresholds_train,
        precision_train[:-1],
        "b--",
        label="Train Precision",
        alpha=0.3,
    )
    plt.plot(
        thresholds_train,
        recall_train[:-1],
        "g--",
        label="Train Recall",
        alpha=0.3,
    )
    plt.plot(
        thresholds_train,
        f1_train[:-1],
        "r--",
        label="Train F1 Score",
        alpha=0.3,
    )

    # Test metrics
    plt.plot(
        thresholds_test,
        precision_test[:-1],
        "b-",
        label="Test Precision",
        alpha=0.9,
    )
    plt.plot(thresholds_test, recall_test[:-1], "g-", label="Test Recall", alpha=0.9)
    plt.plot(thresholds_test, f1_test[:-1], "r-", label="Test F1 Score", alpha=0.9)

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision, Recall, and F1 Score vs. Threshold (Patient Level)")
    plt.legend(loc="best")
    plt.grid(False)
    plt.show()


plot_metrics_vs_threshold_patient(
    y_train_patient, y_train_patient_proba, y_test_patient, y_test_patient_proba
)

# In[110]:


# In[111]:


# In[112]:
# patinets anchor level

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


def calculate_metrics_for_mlflow(
    y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba
):
    """Calculate all classification metrics for MLflow logging."""

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    return (
        train_accuracy,
        train_auc,
        test_accuracy,
        test_auc,
        test_precision,
        test_recall,
        test_f1,
    )


# =============================================================================
# Usage
# =============================================================================

train_accuracy, train_auc, test_accuracy, test_auc, test_precision, test_recall, test_f1 = (
    calculate_metrics_for_mlflow(
        y_train, y_train_pred, y_train_proba, y_test, y_test_pred, y_test_proba
    )
)

# In[113]:

# Patients anchor level.

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score


def precision_recall_at_thresholds(y_train, y_train_proba, y_test, y_test_proba):
    """
    Calculate precision and recall at each threshold (0.01, 0.02, ... 1.00)
    for both training and testing data.
    """

    thresholds = np.arange(0.01, 1.01, 0.01)

    results = []

    for thresh in thresholds:
        # Predictions at threshold
        y_train_pred = (y_train_proba >= thresh).astype(int)
        y_test_pred = (y_test_proba >= thresh).astype(int)

        # Train metrics
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)

        # Test metrics
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)

        results.append(
            {
                "threshold": round(thresh, 2),  # Changed to 2 decimal places
                "train_precision": train_precision,
                "train_recall": train_recall,
                "test_precision": test_precision,
                "test_recall": test_recall,
            }
        )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print(f"{'='*80}")
    print(f"  PRECISION & RECALL AT DIFFERENT THRESHOLDS (0.01 to 1.00)")
    print(f"{'='*80}")
    print(
        f"\n{'Threshold':<12} {'Train Prec':<14} {'Train Recall':<14} {'Test Prec':<14} {'Test Recall':<14}"
    )
    print(f"{'-'*68}")

    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<12.2f} {row['train_precision']:<14.4f} {row['train_recall']:<14.4f} {row['test_precision']:<14.4f} {row['test_recall']:<14.4f}"
        )

    print(f"{'='*80}")
    print(f"  Total Thresholds: {len(results_df)}")
    print(f"{'='*80}")

    return results_df


# =============================================================================
# Usage
# =============================================================================

threshold_results_df = precision_recall_at_thresholds(
    y_train, y_train_proba, y_test, y_test_proba
)

threshold_results_df

# In[114]:
display(threshold_results_df)

# In[115]:
# Patient Level

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score


def get_patient_level_data(y_true, y_proba):
    """Get patient level data using max anchor date's probability."""

    patient_anchor_df = pd.DataFrame(
        {
            "patient_id_anchor": y_true.index,
            "outcome_flag": y_true.values,
            "probability": y_proba,
        }
    )

    patient_anchor_df[["patient_id", "anchor_date"]] = patient_anchor_df[
        "patient_id_anchor"
    ].str.rsplit("_", n=1, expand=True)
    patient_anchor_df["anchor_date"] = pd.to_datetime(patient_anchor_df["anchor_date"])

    patient_level_df = patient_anchor_df.loc[
        patient_anchor_df.groupby("patient_id")["anchor_date"].idxmax()
    ].reset_index(drop=True)

    return patient_level_df["outcome_flag"].values, patient_level_df["probability"].values


def precision_recall_at_thresholds_patient_level(
    y_train, y_train_proba, y_test, y_test_proba
):
    """
    Calculate precision and recall at each threshold (0.01, 0.02, ... 1.00)
    for both training and testing data at PATIENT LEVEL.
    """

    # Get patient level data
    y_train_patient, y_train_patient_proba = get_patient_level_data(
        y_train, y_train_proba
    )
    y_test_patient, y_test_patient_proba = get_patient_level_data(
        y_test, y_test_proba
    )

    print(
        f"Train - Patient Anchor Pairs: {len(y_train)}, Unique Patients: {len(y_train_patient)}"
    )
    print(
        f"Test - Patient Anchor Pairs: {len(y_test)}, Unique Patients: {len(y_test_patient)}"
    )

    thresholds = np.arange(0.01, 1.01, 0.01)

    results = []

    for thresh in thresholds:
        # Predictions at threshold
        y_train_pred = (y_train_patient_proba >= thresh).astype(int)
        y_test_pred = (y_test_patient_proba >= thresh).astype(int)

        # Train metrics
        train_precision = precision_score(y_train_patient, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train_patient, y_train_pred, zero_division=0)

        # Test metrics
        test_precision = precision_score(y_test_patient, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test_patient, y_test_pred, zero_division=0)

        results.append(
            {
                "threshold": round(thresh, 2),
                "train_precision": train_precision,
                "train_recall": train_recall,
                "test_precision": test_precision,
                "test_recall": test_recall,
            }
        )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  PRECISION & RECALL AT DIFFERENT THRESHOLDS - PATIENT LEVEL")
    print(f"{'='*80}")
    print(
        f"\n{'Threshold':<12} {'Train Prec':<14} {'Train Recall':<14} {'Test Prec':<14} {'Test Recall':<14}"
    )
    print(f"{'-'*68}")

    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<12.2f} {row['train_precision']:<14.4f} {row['train_recall']:<14.4f} {row['test_precision']:<14.4f} {row['test_recall']:<14.4f}"
        )

    print(f"{'='*80}")
    print(f"  Total Thresholds: {len(results_df)}")
    print(f"{'='*80}")

    return results_df


# =============================================================================
# Usage
# =============================================================================

threshold_results_patient_df = precision_recall_at_thresholds_patient_level(
    y_train, y_train_proba, y_test, y_test_proba
)

threshold_results_patient_df

# In[116]:
display(threshold_results_patient_df)

# # Check here

# In[117]:


# In[118]:


# In[119]:


# In[120]:
# Train

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def metrics_at_thresholds(y_true, y_proba, dataset_name="Train"):
    """
    Calculate TP, TN, FP, FN, Precision, Recall at each threshold (0.01 to 1.00).

    Args:
        y_true: Actual labels
        y_proba: Predicted probabilities
        dataset_name: Name for the dataset (Train/Test)

    Returns:
        DataFrame with metrics at each threshold
    """

    thresholds = np.arange(0.01, 1.01, 0.01)

    results = []

    for thresh in thresholds:
        # Predictions at threshold
        y_pred = (y_proba >= thresh).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Precision and Recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        results.append(
            {
                "threshold": round(thresh, 2),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
            }
        )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*95}")
    print(f"  {dataset_name.upper()} - METRICS AT DIFFERENT THRESHOLDS")
    print(f"{'='*95}")
    print(
        f"\n{'Threshold':<12} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10} {'Precision':<12} {'Recall':<12}"
    )
    print(f"{'-'*85}")

    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<12.2f} {row['TP']:<10,} {row['TN']:<10,} {row['FP']:<10,} {row['FN']:<10,} {row['precision']:<12.4f} {row['recall']:<12.4f}"
        )

    print(f"{'='*95}")
    print(f"  Total Thresholds: {len(results_df)}")
    print(
        f"  Total Samples: {len(y_true):,} | Positives: {y_true.sum():,} | Negatives: {(y_true == 0).sum():,}"
    )
    print(f"{'='*95}")

    return results_df


# =============================================================================
# Usage
# =============================================================================

# Train metrics
train_threshold_df = metrics_at_thresholds(y_train, y_train_proba, "Train")

# In[121]:
# Test
# Test metrics
test_threshold_df = metrics_at_thresholds(y_test, y_test_proba, "Test")

# In[122]:
display(train_threshold_df)

# In[123]:
display(test_threshold_df)

# In[124]:
# Patient Level

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def get_patient_level_data(y_true, y_proba):
    """Get patient level data using max anchor date's probability."""

    patient_anchor_df = pd.DataFrame(
        {
            "patient_id_anchor": y_true.index,
            "outcome_flag": y_true.values,
            "probability": y_proba,
        }
    )

    patient_anchor_df[["patient_id", "anchor_date"]] = patient_anchor_df[
        "patient_id_anchor"
    ].str.rsplit("_", n=1, expand=True)
    patient_anchor_df["anchor_date"] = pd.to_datetime(patient_anchor_df["anchor_date"])

    patient_level_df = patient_anchor_df.loc[
        patient_anchor_df.groupby("patient_id")["anchor_date"].idxmax()
    ].reset_index(drop=True)

    return patient_level_df["outcome_flag"].values, patient_level_df["probability"].values


def metrics_at_thresholds_patient_level(y_true, y_proba, dataset_name="Train"):
    """
    Calculate TP, TN, FP, FN, Precision, Recall at each threshold (0.01 to 1.00) at PATIENT LEVEL.

    Args:
        y_true: Actual labels (with patient_id_anchor as index)
        y_proba: Predicted probabilities
        dataset_name: Name for the dataset (Train/Test)

    Returns:
        DataFrame with metrics at each threshold
    """

    # Get patient level data
    y_true_patient, y_proba_patient = get_patient_level_data(y_true, y_proba)

    print(f"\n{dataset_name} - Patient Anchor Pairs: {len(y_true)}, Unique Patients: {len(y_true_patient)}")

    thresholds = np.arange(0.01, 1.01, 0.01)

    results = []

    for thresh in thresholds:
        # Predictions at threshold
        y_pred = (y_proba_patient >= thresh).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_true_patient, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Precision and Recall
        precision = precision_score(y_true_patient, y_pred, zero_division=0)
        recall = recall_score(y_true_patient, y_pred, zero_division=0)

        results.append(
            {
                "threshold": round(thresh, 2),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
            }
        )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*95}")
    print(f"  {dataset_name.upper()} - METRICS AT DIFFERENT THRESHOLDS (PATIENT LEVEL)")
    print(f"{'='*95}")
    print(
        f"\n{'Threshold':<12} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10} {'Precision':<12} {'Recall':<12}"
    )
    print(f"{'-'*85}")

    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<12.2f} {row['TP']:<10,} {row['TN']:<10,} {row['FP']:<10,} {row['FN']:<10,} {row['precision']:<12.4f} {row['recall']:<12.4f}"
        )

    print(f"{'='*95}")
    print(f"  Total Thresholds: {len(results_df)}")
    print(
        f"  Total Patients: {len(y_true_patient):,} | Positives (Gold): {y_true_patient.sum():,} | Negatives (Unlabelled): {(y_true_patient == 0).sum():,}"
    )
    print(f"{'='*95}")

    return results_df


# =============================================================================
# Usage
# =============================================================================

# Train metrics - Patient Level
train_threshold_patient_df = metrics_at_thresholds_patient_level(
    y_train, y_train_proba, "Train"
)

# Test metrics - Patient Level
test_threshold_patient_df = metrics_at_thresholds_patient_level(
    y_test, y_test_proba, "Test"
)

# In[125]:
display(train_threshold_patient_df)

# In[126]:
display(test_threshold_patient_df)

# #### Last 2 months GOLD data summary
#
# In[127]:
# Class predictions
y_gold_2_month_pred = model.predict(X_gold_last_2month)
# Probability predictions
y_gold_2month_proba = model.predict_proba(X_gold_last_2month)[:, 1]

# X_gold_last_2month , y_gold_last_2month

# In[128]:
X_gold_last_2month.head(3)

# In[129]:
y_gold_last_2month.head(3)

# In[130]:
# patients anchor level

from sklearn.metrics import recall_score, confusion_matrix, precision_recall_curve
import numpy as np
import pandas as pd

# Overall recall at default threshold (0.5)
recall_default = recall_score(y_gold_last_2month, y_gold_2_month_pred)
print(f"Recall at default threshold (0.5): {recall_default:.4f}")

# Get recall and other metrics at various thresholds
thresholds = np.arange(0.01, 1.01, 0.01)

results = []
for thresh in thresholds:
    y_pred_thresh = (y_gold_2month_proba >= thresh).astype(int)

    cm = confusion_matrix(y_gold_last_2month, y_pred_thresh, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    results.append(
        {
            "threshold": round(thresh, 2),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
        }
    )

results_df = pd.DataFrame(results)
print("\nMetrics at different thresholds:")
display(results_df)

# In[131]:
# Recall at patient level


# Create a dataframe with patient_id, anchor_date, probability and actual outcome
patient_anchor_df = pd.DataFrame(
    {
        "patient_id_anchor": y_gold_last_2month.index,
        "outcome_flag": y_gold_last_2month.values,
        "probability": y_gold_2month_proba,
    }
)

# Split patient_id and anchor_date
patient_anchor_df[["patient_id", "anchor_date"]] = patient_anchor_df[
    "patient_id_anchor"
].str.rsplit("_", n=1, expand=True)
patient_anchor_df["anchor_date"] = pd.to_datetime(patient_anchor_df["anchor_date"])

# Get the max anchor date row for each patient
patient_level_df = patient_anchor_df.loc[
    patient_anchor_df.groupby("patient_id")["anchor_date"].idxmax()
].reset_index(drop=True)

print(f"Total patient-anchor pairs: {len(patient_anchor_df)}")
print(f"Total unique patients: {len(patient_level_df)}")

# Now calculate metrics at patient level with 100 thresholds
thresholds = np.arange(0.01, 1.01, 0.01)

patient_results = []
for thresh in thresholds:
    y_pred_thresh = (patient_level_df["probability"] >= thresh).astype(int)

    # Use labels to ensure 2x2 confusion matrix even if one class is missing
    cm = confusion_matrix(patient_level_df["outcome_flag"], y_pred_thresh, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    patient_results.append(
        {
            "threshold": round(thresh, 2),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4),
        }
    )

patient_results_df = pd.DataFrame(patient_results)
print("\nPatient-level metrics at different thresholds:")
display(patient_results_df)

# In[132]:
10615 + 12005

# #### 4.6 Shap
#
# ##### 4.6.1 Train Shap
#
# In[133]:
import shap
import matplotlib.pyplot as plt

# # Assuming model is your trained model and X is your training data

# # Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Set figure size
plt.figure(figsize=(10, 10))

# SHAP Swarm Plot
shap.summary_plot(shap_values, X_train, max_display=30, plot_size=(22, 12))

# In[134]:
# Shap importance
feature_names = X_train.columns
rf_resultX_train = pd.DataFrame(shap_values, columns=feature_names)

vals = np.abs(rf_resultX_train.values).mean(0)

shap_importance_train = pd.DataFrame(
    list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
)
shap_importance_train.sort_values(
    by=["feature_importance_vals"], ascending=False, inplace=True
)
shap_importance_train["normalized_importance"] = (
    shap_importance_train["feature_importance_vals"]
    / shap_importance_train["feature_importance_vals"].sum()
)
shap_importance_train.head(30)

# In[135]:
display(shap_importance_train)

# ##### 4.6.2 Test Shap
#
# In[136]:
import shap
import matplotlib.pyplot as plt

# # Assuming model is your trained model and X is your training data

# # Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Set figure size
plt.figure(figsize=(10, 10))

# SHAP Swarm Plot
shap.summary_plot(shap_values, X_test, max_display=30, plot_size=(22, 12))

# In[137]:

# Shap importance
feature_names = X_test.columns
rf_resultX_test = pd.DataFrame(shap_values, columns=feature_names)

vals = np.abs(rf_resultX_test.values).mean(0)

shap_importance_test = pd.DataFrame(
    list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
)
shap_importance_test.sort_values(
    by=["feature_importance_vals"], ascending=False, inplace=True
)
shap_importance_test["normalized_importance"] = (
    shap_importance_test["feature_importance_vals"]
    / shap_importance_test["feature_importance_vals"].sum()
)
shap_importance_test.head(30)

# In[138]:
display(shap_importance_test)

# In[139]:
# =====================
# SAVE TO /dbfs/ PATH
# =====================

# Define DBFS path (use /dbfs/ prefix for pandas)
# output_path_shap_train = "/dbfs/FileStore/xgb_classification_shap_train_iteration_2.csv"
output_path_shap_test = "/dbfs/FileStore/xgb_classification_shap_test_iteration_2.csv"

# Save Train SHAP
# shap_importance_train.to_csv(output_path_shap_train, index=False)
# print(f"✅ SHAP Train saved to: {output_path_shap_train}")
# print(f"   Shape: {shap_importance.shape}")

# # Save Test SHAP
shap_importance_test.to_csv(output_path_shap_test, index=False)
print(f"✅ SHAP Test saved to: {output_path_shap_test}")
print(f"   Shape: {shap_importance_test.shape}")

# In[140]:


# ### 5. Model Export - Unity Catalog
#
# #### 5.1 Catalog and Schema Name
#
# In[141]:
# Get current catalog
current_catalog = spark.sql("SELECT current_catalog()").collect()[0][0]
print(f"\nCurrent Catalog: {current_catalog}")

# Get current schema/database
current_schema = spark.sql("SELECT current_schema()").collect()[0][0]
print(f"Current Schema: {current_schema}")

# #### 5.2 Parameters
#
# In[142]:
import mlflow
import mlflow.spark
import mlflow.xgboost

# In[143]:
# =============================
# SAVE MODELS TO UNITY CATALOG
# =============================

CATALOG = current_catalog  # "hive_metastore"
SCHEMA = current_schema  # "default"

# Model names
XGB_MODEL_NAME = "xgboost_classifier_model_iteration_1_patients_split_freq_c_freq"

# Full registered model paths

REGISTERED_XGB_MODEL = f"{CATALOG}.{SCHEMA}.{XGB_MODEL_NAME}"

# Set registry URI for Unity Catalog
mlflow.set_registry_uri("databricks-uc")

print("SAVING MODELS TO UNITY CATALOG")
print(f"\nCatalog: {CATALOG}")
print(f"Schema: {SCHEMA}")

print(f"XGB Model Path: {REGISTERED_XGB_MODEL}")

# #### 5.3 Saving the Models
#
# In[144]:
# ============================================================================
# SAVE MODELS TO DATABRICKS (FEATURE SELECTION)
# ============================================================================

import mlflow
import mlflow.spark
import mlflow.xgboost
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# ============================================================================
# 🔧 CONFIG
# ============================================================================

# End any existing runs
mlflow.end_run()

# Set tracking URI to Databricks
mlflow.set_tracking_uri("databricks")

# Set registry URI to Databricks
mlflow.set_registry_uri("databricks")

# Set experiment
mlflow.set_experiment("/Shared/mash_model_experiments_classification")

XGB_MODEL_NAME = XGB_MODEL_NAME

# ============================================================================
# 📌 SAVE XGBOOST MODEL
# ============================================================================

print("\n[Saving XGBoost Model...]")
with mlflow.start_run(run_name="XGBoost_Model_Classification"):
    # Log parameters
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("purpose", "Model Iteration 1")
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.log_param("max_depth", best_params["max_depth"])
    mlflow.log_param("learning_rate", best_params["learning_rate"])
    mlflow.log_param("subsample", best_params["subsample"])

    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("train_auc", train_auc)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)

    # Log model
    mlflow.xgboost.log_model(
        model,
        "xgb_model",
        registered_model_name=XGB_MODEL_NAME,
    )
    print(f"✅ XGBoost Model saved as: {XGB_MODEL_NAME}")

print(
    f"""
Model saved to Databricks Model Registry:
  - XGBoost Model: {XGB_MODEL_NAME}

Purpose: Model Iteration

To view: Go to Databricks UI → Machine Learning → Models
"""
)

# #### 5.4 Loading the Saved Model
#
# In[145]:
# ============================================================================
# LOAD FEATURE SELECTION MODELS FROM DATABRICKS (FIXED)
# ============================================================================

import mlflow
import mlflow.spark
import xgboost as xgb
import os

# Set URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# Model names

XGB_MODEL_NAME = (
    "xgboost_classifier_model_iteration_1_patients_split_freq_c_freq"  # XGB_MODEL_NAME
)

# ============================================================================
# 📌 LOAD XGBOOST MODEL (FIXED - Download and Load Manually)
# ============================================================================

print("\nLoading XGBoost Model (Feature Selection)...")

# Step 1: Get the model version info
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions(XGB_MODEL_NAME, stages=["None"])[0]
run_id = model_version.run_id

print(f"Run ID: {run_id}")

# Step 2: Download the model artifact
artifact_path = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="xgb_model"
)
print(f"Artifact downloaded to: {artifact_path}")

# Step 3: Find the model file
model_file = None
for file in os.listdir(artifact_path):
    if file.endswith(".json") or file.endswith(".xgb") or file == "model.xgb":
        model_file = os.path.join(artifact_path, file)
        break

# If model.xgb not found, try model subdirectory
if model_file is None:
    model_subdir = os.path.join(artifact_path, "model.xgb")
    if os.path.exists(model_subdir):
        model_file = model_subdir

print(f"Model file: {model_file}")

# Step 4: Load as XGBoost Booster
loaded_xgb_booster = xgb.Booster()
loaded_xgb_booster.load_model(model_file)

print(f"✅ XGBoost Model loaded!")

# ============================================================================
# 📌 TEST PREDICTIONS
# ============================================================================

print("\n[Testing XGBoost Predictions]")

# Create DMatrix for prediction
dtest = xgb.DMatrix(X_test.head(5))

# Predict probabilities
sample_pred_proba = loaded_xgb_booster.predict(dtest)

# Convert to binary predictions
sample_pred = (sample_pred_proba > 0.5).astype(int)

print(f"Sample Probabilities: {sample_pred_proba}")
print(f"Sample Predictions: {sample_pred}")

print("\n✅ Feature Selection Models loaded and working!")

# In[146]:


# ### 6. Prediction on Unlabelled data.
#
# #### 6.1 MASH
#
# In[147]:
mash_gold_unlabelled_selected_feat_with_outcome_df.head()

# In[148]:
mash_gold_unlabelled_selected_feat_with_outcome_df["outcome_flag"].value_counts()

# In[149]:
mash_unlabelled_for_prediction = mash_gold_unlabelled_selected_feat_with_outcome_df[
    mash_gold_unlabelled_selected_feat_with_outcome_df["outcome_flag"] == 0
]
mash_unlabelled_for_prediction.shape

# In[150]:
# ============================================================================
# MASH UNLABELLED PREDICTION AND SUMMARY
# ============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# ============================================================================
# 1. PREPARE DATA AND PREDICT
# ============================================================================

print("=" * 80)
print("MASH UNLABELLED PREDICTION")
print("=" * 80)

# Prepare features (exclude patient_id and outcome_flag)
feature_cols = [
    col for col in mash_unlabelled_for_prediction.columns if col not in ["patient_id", "outcome_flag"]
]

X_unlabelled = mash_unlabelled_for_prediction[feature_cols]
X_unlabelled.index = mash_unlabelled_for_prediction["patient_id"]

# Create DMatrix for prediction
d_unlabelled = xgb.DMatrix(X_unlabelled)

# Predict probabilities
unlabelled_pred_proba = loaded_xgb_booster.predict(d_unlabelled)

# Predict classes at default threshold (0.5)
unlabelled_pred = (unlabelled_pred_proba > 0.5).astype(int)

print(f"\nTotal Unlabelled Records: {len(X_unlabelled):,}")
print(
    f"Unique Patients: {mash_unlabelled_for_prediction['patient_id'].str.rsplit('_', n=1).str[0].nunique():,}"
)

# In[151]:
# ============================================================================
# 4. PREDICTIONS AT DIFFERENT THRESHOLDS
# ============================================================================


def predictions_at_thresholds_unlabelled(y_proba):
    """
    Calculate predicted positives at each threshold (0.01 to 1.00).
    """

    thresholds = np.arange(0.01, 1.01, 0.01)

    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        predicted_positive = y_pred.sum()
        predicted_negative = (y_pred == 0).sum()
        positive_rate = predicted_positive / len(y_pred) * 100

        results.append(
            {
                "threshold": round(thresh, 2),
                "predicted_positive": predicted_positive,
                "predicted_negative": predicted_negative,
                "positive_rate": round(positive_rate, 2),
            }
        )

    results_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"  PREDICTIONS AT DIFFERENT THRESHOLDS")
    print(f"{'='*70}")
    print(
        f"\n{'Threshold':<12} {'Pred Positive':<18} {'Pred Negative':<18} {'Positive Rate':<15}"
    )
    print(f"{'-'*60}")

    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<12.2f} {row['predicted_positive']:<18,} {row['predicted_negative']:<18,} {row['positive_rate']:<15.2f}%"
        )

    print(f"{'='*70}")

    return results_df


unlabelled_threshold_df = predictions_at_thresholds_unlabelled(unlabelled_pred_proba)
display(unlabelled_threshold_df)

# In[152]:
# Create result dataframe with patient_id and predictions
unlabelled_result_df = pd.DataFrame(
    {
        "patient_id": mash_unlabelled_for_prediction["patient_id"],
        "pred_proba": unlabelled_pred_proba,
        "pred_class_0.5": unlabelled_pred,
    }
)

# Extract patient_id only (without anchor date)
unlabelled_result_df["pat_id"] = unlabelled_result_df["patient_id"].str.rsplit(
    "_", n=1
).str[0]
unlabelled_result_df["anchor_date"] = unlabelled_result_df["patient_id"].str.rsplit(
    "_", n=1
).str[1]

unlabelled_result_df.head()

# #### 6.2 MASLD prediction
#
#

# In[153]:
display(parq_masld_gold_unlabelled_with_outcome_data.limit(10))

# In[154]:
parq_masld_unlabelled_with_outcome_data = parq_masld_gold_unlabelled_with_outcome_data.filter(
    col("outcome_flag") == 0
)
display(parq_masld_unlabelled_with_outcome_data.limit(10))

# In[155]:
masld_unlabelled_with_outcome_df = parq_masld_unlabelled_with_outcome_data.toPandas()
masld_unlabelled_with_outcome_df.head()

# In[156]:
# ============================================================================
# MASLD UNLABELLED PREDICTION AND SUMMARY
# ============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb

# ============================================================================
# 1. PREPARE DATA AND PREDICT
# ============================================================================

print("=" * 80)
print("MASLD UNLABELLED PREDICTION")
print("=" * 80)

# Prepare features (exclude patient_id and outcome_flag)
feature_cols = [
    col for col in masld_unlabelled_with_outcome_df.columns if col not in ["patient_id", "outcome_flag"]
]

X_masld_unlabelled = masld_unlabelled_with_outcome_df[feature_cols]
X_masld_unlabelled.index = masld_unlabelled_with_outcome_df["patient_id"]

# Create DMatrix for prediction
d_masld_unlabelled = xgb.DMatrix(X_masld_unlabelled)

# Predict probabilities
masld_unlabelled_pred_proba = loaded_xgb_booster.predict(d_masld_unlabelled)

# Predict classes at default threshold (0.5)
masld_unlabelled_pred = (masld_unlabelled_pred_proba > 0.5).astype(int)

print(f"\nTotal Unlabelled Records: {len(X_masld_unlabelled):,}")
print(
    f"Unique Patients: {masld_unlabelled_with_outcome_df['patient_id'].str.rsplit('_', n=1).str[0].nunique():,}"
)

# In[157]:
# ============================================================================
# 4. PREDICTIONS AT DIFFERENT THRESHOLDS
# ============================================================================


def predictions_at_thresholds_unlabelled(y_proba, dataset_name="MASLD"):
    """
    Calculate predicted positives at each threshold (0.01 to 1.00).
    """

    thresholds = np.arange(0.01, 1.01, 0.01)

    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        predicted_positive = y_pred.sum()
        predicted_negative = (y_pred == 0).sum()
        positive_rate = predicted_positive / len(y_pred) * 100

        results.append(
            {
                "threshold": round(thresh, 2),
                "predicted_positive": predicted_positive,
                "predicted_negative": predicted_negative,
                "positive_rate": round(positive_rate, 2),
            }
        )

    results_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"  {dataset_name} - PREDICTIONS AT DIFFERENT THRESHOLDS")
    print(f"{'='*70}")
    print(
        f"\n{'Threshold':<12} {'Pred Positive':<18} {'Pred Negative':<18} {'Positive Rate':<15}"
    )
    print(f"{'-'*60}")

    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<12.2f} {row['predicted_positive']:<18,} {row['predicted_negative']:<18,} {row['positive_rate']:<15.2f}%"
        )

    print(f"{'='*70}")

    return results_df


masld_unlabelled_threshold_df = predictions_at_thresholds_unlabelled(
    masld_unlabelled_pred_proba, "MASLD"
)

# In[158]:
display(masld_unlabelled_threshold_df)

# In[159]:
# ============================================================================
# 5. CREATE PREDICTION RESULT DATAFRAME
# ============================================================================

# Create result dataframe with patient_id and predictions
masld_unlabelled_result_df = pd.DataFrame(
    {
        "patient_id": masld_unlabelled_with_outcome_df["patient_id"],
        "pred_proba": masld_unlabelled_pred_proba,
        "pred_class_0.5": masld_unlabelled_pred,
    }
)

# Extract patient_id only (without anchor date)
masld_unlabelled_result_df["pat_id"] = masld_unlabelled_result_df["patient_id"].str.rsplit(
    "_", n=1
).str[0]
masld_unlabelled_result_df["anchor_date"] = masld_unlabelled_result_df["patient_id"].str.rsplit(
    "_", n=1
).str[1]

masld_unlabelled_result_df.head()

# In[160]:


# In[161]:


# In[162]:

mash_gold_unlabelled_selected_feat_with_outcome_df
parq_masld_gold_with_outcome_data = parq_masld_gold_unlabelled_with_outcome_data.filter(
    col("outcome_flag") == 1
)
