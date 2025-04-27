from app import app, db, Menu, OperatingHours, Reservations, Locations
from datetime import time, datetime

with app.app_context():
    # Drop and recreate all tables
    db.drop_all()
    db.create_all()

    # ====================
    # Add Menu Items
    # ====================
    menu_items = [
        Menu(name="Margherita Pizza", category="Main Course", price=850, is_special=False, description="Classic cheese and tomato pizza."),
        Menu(name="Pepperoni Pizza", category="Main Course", price=950, is_special=True, description="Spicy pepperoni with mozzarella."),
        Menu(name="Caesar Salad", category="Starter", price=500, is_special=False, description="Fresh romaine with Caesar dressing."),
        Menu(name="Chocolate Lava Cake", category="Dessert", price=400, is_special=True, description="Warm chocolate cake with molten center.")
    ]
    db.session.add_all(menu_items)

    # ====================
    # Add Operating Hours
    # ====================
    operating_hours = [
        OperatingHours(day="Monday", open_time=time(10, 0), close_time=time(22, 0)),
        OperatingHours(day="Tuesday", open_time=time(10, 0), close_time=time(22, 0)),
        OperatingHours(day="Wednesday", open_time=time(10, 0), close_time=time(22, 0)),
        OperatingHours(day="Thursday", open_time=time(10, 0), close_time=time(22, 0)),
        OperatingHours(day="Friday", open_time=time(10, 0), close_time=time(23, 0)),
        OperatingHours(day="Saturday", open_time=time(10, 0), close_time=time(23, 0)),
        OperatingHours(day="Sunday", open_time=time(11, 0), close_time=time(21, 0)),
    ]
    db.session.add_all(operating_hours)

    # ====================
    # Add Location Info
    # ====================
    location = Locations(
        address="123 Food Street, Flavor Town",
        phone="+1234567890",
        email="info@flavortown.com"
    )
    db.session.add(location)

    # ====================
    # Add Dummy Reservation
    # ====================
    reservation = Reservations(
        name="Bilal",
        phone="03001234567",
        people_count=2,
        reservation_time=datetime(2025, 4, 25, 19, 30),
        notes="Window seat preferred"
    )
    db.session.add(reservation)

    # Commit all changes
    db.session.commit()
    print("✅ Database seeded successfully!")
