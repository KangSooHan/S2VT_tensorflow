## S2VT_tensorflow

## Environment
  - Python3.6
  - Tensorflow 1.8.0
  
## Directory Tree
  S2VT_tensorflow
  
  ![alt tag](https://github.com/KangSooHan/S2VT_tensorflow/blob/master/Directory.png)
  
## How To Run
  1. Download MSVD dataset in data/youtube_videos
 
     [Dataset](http://www.cs.utexas.edu/users/ml/clamp/videoDescription)
  
  2. Extract Video Features
     ```bash
     $ python extract_RGB_feature.py
     ```
  3. Train model
  
     - run ipython
       ```bash
       $ CUDA_VISIBLE_DEVICES=0 ipython
       ```
       
     ```bash  
     >>> import model_RGB
     
     >>> model_RGB.train()
     ```

  4. Test model
  
     - run ipython
       ```bash
       $ CUDA_VISIBLE_DEVICES=0 ipython
       ```
     ```bash  
     >>> import model_RGB
     
     >>> model_RGB.test()
     ```
     
     You can change save model
     
     
  5. Evaluate with COCO
  
     - move S2VT_Description.txt, S2VT_results.txt to ./caption-eval
       ```bash
       mv S2VT_Description.txt, S2VT_results.txt to ./caption-eval
     
       $python create_json_references.py -i S2VT_Description.txt -o S2VT_Description.json
    
       $python run_evaluations.py -i S2VT_results.txt -r S2VT_Description.json
       ```

## References
  S2VT model by [chenxinpeng](https://github.com/chenxinpeng/S2VT)
  
  Vgg model by [AlonDaks/tsa-kaggle](https://github.com/AlonDaks/tsa-kaggle)
 
