import TrainingModel, Utils
import tensorflow as tf
import transformers
import numpy as np
import tqdm
import pickle
import time

data_path='/home/yuanhuaying/pycharm_project/data/'

# 这个pkl中包含的 English text-image对是中文zh.txt的超集

def load_data():

    dict_path=data_path+'extracted_data.pkl'
    text_en_path=data_path+'text/en.txt'
    text_zh_path=data_path+'text/zh.txt'

    with open(dict_path, 'rb') as f:  # 读取pkl文件数据
        dict_data = pickle.loads(f.read())

    with open(text_en_path,'r') as f:
        text_en_data=f.readlines()
        for i in range(len(text_en_data)):
            text_en_data[i]=text_en_data[i].strip()

    with open(text_zh_path,'r') as f:
        text_zh_data=f.readlines()
        for i in range(len(text_zh_data)):
            text_zh_data[i]=text_zh_data[i].strip()

    # pre/zh.txt存放英文短语，post/zh.txt存放中文短语
    # 根据pre/zh.txt中的英文取出对应的embeddings

    emeddings=[]

    for key in text_en_data:
        if len(dict_data[key])==2:
            emeddings.append(dict_data[key]['Embedding'])
        elif len(dict_data[key])==640:
            emeddings.append(dict_data[key])

    assert len(emeddings) == len(text_zh_data)

    return text_zh_data,emeddings

def prepareDataset(tokenizer, numValidationSamples):
    # 建立文本与对应的feature之间的映射

    # Step1：先把英文语料库的数据翻译成中文
    # Step2：把用英文语料库对应的嵌入特征和中文一一对应，训练语言模型

    # This part you need to prepare yourself!
    # What is needed here is a list of sentences in whatever language(s) you are interested in
    # and a matching set of Clip-Text encoder embeddings for the English counter part.

    # Pre-computed CLIP-Text encoder embeddings for 2 Million images, can be found here:
    # https://drive.google.com/drive/folders/1I9a7naSZubUATWzLFv61DQMWyFlF7wR5

    sentences,emeddings=load_data()

    print("Number of total training samples:", len(sentences))

    inSents, embs = shuffleData(sentences, emeddings)  # Shuffle before selecting validation data
    trainSents, trainEmbs = inSents[numValidationSamples:], embs[numValidationSamples:]
    evalSents, evalEmbs = inSents[:numValidationSamples], embs[:numValidationSamples]
    evalIds, evalAtt = Utils.batchEncode(evalSents, tokenizer)
    evalInData, evalLabels = (evalIds, evalAtt), tf.convert_to_tensor(evalEmbs, tf.float32)
    print("Number of training samples:", len(trainSents))
    print("Number of validation samples:", len(evalSents))

    return trainSents, trainEmbs, evalInData, evalLabels


def shuffleData(sents, embs):
    shuffleOrder = np.random.choice(range(len(sents)), len(sents), replace=False)
    f = lambda x: [x[i] for i in shuffleOrder]
    return f(sents), f(embs)


def createModel(modelBase, clipEmbeddingSize):
    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, clipEmbeddingSize)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelBase)
    return model, tokenizer

def time_cal(s,c,i,e):


def trainStudentTextEncoder():
    modelBase = 'distilbert-base-multilingual-cased'
    numValidationSamples = 2000
    clipEmbeddingSize = 640
    learningRate = 5e-5
    batchSize = 64
    epochs = 100
    fetchSize = 500 * batchSize

    model, tokenizer = createModel(modelBase, clipEmbeddingSize)
    trainSents, trainEmbs, evalIn, evalLabels = prepareDataset(tokenizer, numValidationSamples)

    optim = tf.optimizers.Adam(learningRate)
    model.compile(optim, 'mse', metrics=['mae'])
    saveName = "CLIP-Text-Encoder"


    start_time=time.clock()
    fetchCounter = 0
    for e in range(epochs):
        shuffleData(trainSents, trainEmbs)
        # tqdm.tqdm用来绘制进度条
        for i in tqdm.tqdm(range(0, len(trainSents), fetchSize), desc="Fetches"):
            batchEmbs = tf.convert_to_tensor(trainEmbs[i:i + fetchSize], tf.float32)
            batchSents = trainSents[i:i + fetchSize]

            # tokenizer相当于对文本划分词块，eg "我爱北京天安门“->我爱 北京 天 安 门

            inData = Utils.batchEncode(batchSents, tokenizer)

            model.fit(inData, batchEmbs, batch_size=batchSize, verbose=1,
                      validation_data=(evalIn, evalLabels), shuffle=True)

            fetchCounter += 1
            if (fetchCounter % 50 == 0):
                model.save_weights("{}-{}-Weights".format(saveName, fetchCounter))
                cur_time=time.clock()
                print("Total Time Left: ",time_cal(start_time,cur_time,e,i))


if __name__ == '__main__':
    CUDA_VISIVLE_DEVICES=1
    trainStudentTextEncoder()


