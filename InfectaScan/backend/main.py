from flask import Flask

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello, World!"

@app.route('/runmodel', methods=['GET'])
def runmodel():
    return 



if __name__ == '__main__':
    app.run(host='localhost', port=5000)
