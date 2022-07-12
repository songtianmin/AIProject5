# 人工智能实验五

Project5 for Artificial Intelligence course.

**Tips: GPU can make programs run faster**

## Setup

This implemetation is based on `Python3`.
To run the code, you need the following dependencies:

- torch==1.10.2
- torchvision==0.11.3
- transformers==4.19.2
- numpy==1.22.2
- scikit_learn==1.1.1
- Pillow==9.2.0
- tqdm==4.62.3

You can run

```
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.	

```
├-dataset
│  │--data		# text and image data 
│     │-- xx.jpg and xx.txt
│  │--test_without_label.txt	# test data for predict
│  │--train.txt		 # train data before split,used to train the final model  		    
├-pretrained_bert_mode  ls
│  │--bert-base-uncased/    # pretrained model bert-base-uncased
│      │-- config.json    
│      │-- pytorch_model.bin
│      │-- vocab.txt
├-train_model
│  │--best-model.pth		# best model saved during training with train type 'dev'
│  │--result.txt                # prediction results of best model
│--dataset.py		# code for create dataset
│--main.py		# main code for calling other parts
│--model.py		# code for model(VistaNet)
│--model2.py            # code for model2
│--README.md
│--requirements.txt
│--train_and_eval.py	# code for train,test and verify,predict 
```

## Run code

To run code, the template of the script running on cmd is as follows

```
python main.py [args]
```

args(you can run `python main.py --help` to see the usage):

```
-h, --help                        show this help message and exit
-do_train		          train or not
-do_test			  test or not (tips:during testing, training and verificattion are not supported)
-do_eval_ablation	          dev or not (need -ablation)
-ablation	                  whether ablation experiment verification is used(0: not used,1: text,2: image) 
-epoch             	          train epoch num
-batch_size                       batch size number
-lr                               learning rate
```

We provide 3 different `operation modes`. You can choose one according to the following description.

- `train`

  In this operation mode,the code will divide original train data into train data and dev data,then model will be trained on train data and show the result on dev data.Finally,the code will save the best model in `/train_model/best-model.pth`.
  
  To run this operation mode,you can run following script on cmd
  
  ```
  python main.py -do_train True
  ```


- `test`

  In this operation mode,the code will **predict the test data** with the model saved in `/train_model/best_model.pth`,and save the results in `/train_model/result.txt`.

  To run this operation mode,you can run following script on cmd:

  ```
  python main.py -do_test True
  ```

- `dev`

    In this operation mode,the code will **verification the eval data (whether ablation)** with the model saved in `/train_model/best_model.pth`.
To run this operation mode,you can run following script on cmd:
    - `non ablation experiment`
    ```
    python main.py -do_eval_ablation -ablation 0
    ```
  - `text`
      
  ```
  python main.py -do_eval_ablation -ablation 1
  ```
  - `image`
      
  ```
  python main.py -do_eval_ablation -ablation 0
  ```


## Attribution

Parts of this code are based on the following repositories:

- [VistaNet](https://ojs.aaai.org/index.php/AAAI/article/view/3799)