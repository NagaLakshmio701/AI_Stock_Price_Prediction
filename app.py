from flask import Flask, render_template, request
from utils import predict_stock


app = Flask(__name__)
# Health check route for Azure
@app.route('/robots933456.txt')
def health_probe():
    return 'OK', 200

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        days = int(request.form.get('days'))

        prediction = predict_stock(ticker, days)

    prediction_data = prediction.to_dict(orient='records') if prediction is not None else None
    return render_template('index.html', prediction=prediction_data)

if __name__ == '__main__':
    app.run(debug=True)
