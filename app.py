from flask import Flask,request,render_template,redirect

app = Flask(__name__)

name = 0
movies = [
    {'title': 'My Neighbor Totoro', 'year': '1988'},
    {'title': 'Dead Poets Society', 'year': '1989'},
    {'title': 'A Perfect World', 'year': '1993'},
    {'title': 'Leon', 'year': '1994'},
    {'title': 'Mahjong', 'year': '1996'},
    {'title': 'Swallowtail Butterfly', 'year': '1996'},
    {'title': 'King of Comedy', 'year': '1999'},
    {'title': 'Devils on the Doorstep', 'year': '1999'},
    {'title': 'WALL-E', 'year': '2008'},
    {'title': 'The Pork of Music', 'year': '2012'},
]


@app.route("/",methods=['GET','POST'])
def login():
    if request.method =='POST':
        username = request.form['username']
        if username =="user":
            return render_template('login1.html', name=1, movies=movies)
        else:
            message = "Failed Login"
            return render_template('login1.html',name=0)
    return render_template('login1.html',name=0)

if __name__ == '__main__':
    app.run(debug=True)
