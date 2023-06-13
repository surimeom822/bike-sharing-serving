import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weather",
]  # ["time", "season", "holiday", "workingday", "weather"]


def extract_time(time_info: str) -> str:
    time_info = time_info.strftime("%Y-%m-%d %H:%M:%S")

    split_time_info = time_info.split(" ")[1].split(":")
    time_str = split_time_info[0]
    return time_str


def time_extractor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df["time"] = df[col].apply(lambda x: extract_time(x))
    return df


preprocess_pipeline = ColumnTransformer(
    transformers=[
        # (
        #     "time_extractor",
        #     FunctionTransformer(
        #         time_extractor, kw_args={"col": "datetime"}
        #     ),
        #     ["datetime"],
        # ),
        (
            "target_encoder",
            TargetEncoder(),
            CAT_FEATURES,
        ),
    ],
    remainder="passthrough",  # default="drop" 설정 시 변환되지 않는 변수는 삭제됨
    verbose_feature_names_out=False,  # 원래 변수를 덮어씌움 (True일 경우 transformer이름+column이름을 가진 변수 생성)
)
preprocess_pipeline.set_output(
    transform="pandas"
)  # transform 지정하지 않을 경우 np.ndarray 반환
