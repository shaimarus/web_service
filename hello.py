from flask import Flask,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)


model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return 'Hello, World! sdfsdfsd 1111 sdfs'  


@app.route('/mnt/<nanan>')
def hello_world2(nanan):
    return 'Hello, World! sdfsdfsd 1111 sdfs2 222 333'  + str(nanan)

@app.route('/mnt2/<nanan>')
def hello_world3(nanan):
    return str(float(nanan)+float(nanan))

@app.route('/avg/<x>')
def hello_world4(x):
    x=x.split(',')
    x=[float(i) for i in x]
    return str(np.mean(x))
@app.route('/predict/<x>')
def predict(x):
    x=x.split(',')
    x=[float(i) for i in x]
    x=np.array(x).reshape(1,-1)  
    return str('xgboost classifier iris:{}'.format(model.predict(x)))

@app.route('/iris_post',methods=['POST'])
def dd():
    content=request.get_json()

    x=content['flower'].split(',')
    x=[float(i) for i in x]
    x=np.array(x).reshape(1,-1)  
    predict=model.predict(x)

    return jsonify(str(predict[0]))

print(1+3)
print(1-1)





