from flask import Flask, render_template



app = Flask(__name__, template_folder='views')



@app.route('/')
def home():

    return render_template('home.html')

@app.route('/about.html')
def about():

    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5110)
