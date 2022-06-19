## competition_NIPA_AI_2022 : https://aichallenge.or.kr/
+ https://www.tensorflow.org/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa
+ https://www.tensorflow.org/text/tutorials/uncertainty_quantification_with_sngp_bert
+ http://aiopen.etri.re.kr/aidata_download.php

### 사용 설명서 
+ 주 코드 폴더 : ./models/baseline
+ 주 deeplearning Platform : pytorch 1.9
+ dataset
    + /datasets/train.json
    + /datasets/test.json
+ train
    + modify configuretion files
        + ./config/train_config.yaml
          dataset: path_to_train.json  
             --> dataset: /Users/.../competition_NIPA_AI_2022/datasets/ # absolute path
    + python ./train.py 
    + take time : 2 days 13 hours
    + result folder : ./models/baseline/result/train
    
+ predict
    + modify configuretion files
        + ./config/predict_config.yaml
          dataset: path_to_test.json_and_sample_submission  
             --> dataset: /Users/.../competition_NIPA_AI_2022/datasets/ # absolute path
          train_serial: "YYYYMMDD_SERIAL"
            --> train_serial: "20220612_115012"     # ./results/train/20220612_115012/
    + python ./predict.py 
    + result folder : ./models/baseline/result/predict/*

+ submission
    + copy and rename
        + copy ./predict/20220612_115012/prediction.csv
        + rename prediction.csv to submission.csv