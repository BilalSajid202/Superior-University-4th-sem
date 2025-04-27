from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///restaurant.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ========== Database Models ==========
class Menu(db.Model):
    item_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    category = db.Column(db.String(50))
    price = db.Column(db.Numeric(6, 2))
    is_special = db.Column(db.Boolean)
    description = db.Column(db.Text)

class OperatingHours(db.Model):
    day = db.Column(db.String(10), primary_key=True)
    open_time = db.Column(db.Time)
    close_time = db.Column(db.Time)

class Reservations(db.Model):
    reservation_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    phone = db.Column(db.String(15))
    people_count = db.Column(db.Integer)
    reservation_time = db.Column(db.DateTime)
    notes = db.Column(db.Text)

class Locations(db.Model):
    location_id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(255))
    phone = db.Column(db.String(15))
    email = db.Column(db.String(100))

# ========== Chatbot Logic ==========
def restaurant_bot_response(user_input):
    user_input = user_input.lower()

    if "menu" in user_input or "food" in user_input:
        items = Menu.query.all()
        if not items:
            return "Sorry, the menu is currently unavailable."
        return "Here's our menu: " + ", ".join([item.name for item in items])

    elif "special" in user_input or "deal" in user_input:
        specials = Menu.query.filter_by(is_special=True).all()
        if not specials:
            return "No specials today."
        return "Today's specials are: " + ", ".join([item.name for item in specials])

    elif "opening" in user_input or "timing" in user_input or "hours" in user_input or "open" in user_input:
        hours = OperatingHours.query.all()
        if not hours:
            return "Operating hours are not set."
        return "Our opening hours are:\n" + "\n".join(
            [f"{h.day}: {h.open_time.strftime('%I:%M %p')} - {h.close_time.strftime('%I:%M %p')}" for h in hours]
        )

    elif "location" in user_input or "address" in user_input or "where" in user_input:
        location = Locations.query.first()
        if not location:
            return "Our location details are currently unavailable."
        return f"Our address is {location.address}. You can reach us at {location.phone}."

    elif "reservation" in user_input or "book" in user_input or "table" in user_input:
        return "To make a reservation, please call us or use our online booking system. You can also visit the restaurant directly."

    elif "help" in user_input or "options" in user_input or "what can you do" in user_input:
        return (
            "I can help you with:\n"
            "- 📋 Viewing the menu\n"
            "- ⭐ Today's specials\n"
            "- 🕒 Restaurant hours\n"
            "- 📍 Our location\n"
            "- 📞 Making a reservation\n"
            "Just ask me anything!"
        )

    else:
        return (
            "I'm here to assist you with restaurant info! 😊\n"
            "Try asking things like:\n"
            "• What's on the menu?\n"
            "• Do you have any specials?\n"
            "• When are you open?\n"
            "• Where are you located?\n"
            "• Can I book a table?"
        )

# ========== Routes ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = restaurant_bot_response(user_input)
    return jsonify({'reply': response})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
