from flask import Flask,request , jsonify 

# from model import predict_sentiment

app = Flask(__name__)

@app.route("/user/<userId>")
def getUser(userId):
    return f'user-id : {userId}'

@app.route("/getDetails",methods=["POST"])
def getDetails():
    data=request.get_json()
    if not data or 'statement' not in data:
        return "Yo man, put the sentence!", 400

    return f'the sentence {data.get('user')}  gave is :\n {data.get('statement')} '


# @app.route('/predict',methods=['POST'])
# def predictAPI():
#     try:
#         data = request.get_json()
#         if not data or 'text' not in data: return jsonify({'error': 'Invalid input. Please provide text to analyze.'}), 400
#         text = data['text']
#         sentiment_label = predict_sentiment(text)
#         sentiment_mapping = {1: "positive", 0: "neutral", -1: "negative"}
#         sentiment_text = sentiment_mapping[sentiment_label]

#         return jsonify({
#             'text': text,
#             'sentiment': sentiment_text
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500



if __name__ == '__main__' :
    app.run(debug=True)
