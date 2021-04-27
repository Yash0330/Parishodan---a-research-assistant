from flask import Blueprint, render_template

second=Blueprint("second", __name__, static_folder="static", template_folder="templates")

@second.route("/index")
@second.route("/")
def index():
    return render_template("index.html")

@second.route("/about")
def about():
    return render_template("about.html")

@second.route("/blog")
def blog():
    return render_template("blog.html")

@second.route("/contact")
def contact():
    return render_template("contact.html")

@second.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

@second.route("/service")
def service():
    return render_template("service.html")

@second.route("/single")
def single():
    return render_template("single.html")

@second.route("/team")
def team():
    return render_template("team.html")