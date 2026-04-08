import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stocksmart-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocksmart.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ------------------- MODELS -------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    shop_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    tin = db.Column(db.String(50), nullable=True)
    is_vat_registered = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)   # added for admin features
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    cost_price = db.Column(db.Float, nullable=False)
    selling_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, default=0)
    low_stock_threshold = db.Column(db.Integer, default=5)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    selling_price_at_time = db.Column(db.Float, nullable=False)
    cost_price_at_time = db.Column(db.Float, nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    profit = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StockHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    change_type = db.Column(db.String(20))
    quantity = db.Column(db.Integer)
    note = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Counter model for total visits
class SiteStat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Integer, default=0)

# ------------------- HELPERS -------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access denied. Admin only.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

# ------------------- VISITOR COUNTER (before each request) -------------------
@app.before_request
def count_visitors():
    # skip static files and favicon
    if request.endpoint and 'static' in request.endpoint:
        return
    if request.path == '/favicon.ico':
        return
    stat = SiteStat.query.filter_by(name='total_visitors').first()
    if not stat:
        stat = SiteStat(name='total_visitors', value=0)
        db.session.add(stat)
        db.session.commit()
    stat.value += 1
    db.session.commit()

@app.context_processor
def inject_globals():
    stat = SiteStat.query.filter_by(name='total_visitors').first()
    total_visits = stat.value if stat else 0
    return dict(total_visits=total_visits)

# ------------------- ROUTES -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        shop_name = request.form['shop_name']
        phone = request.form['phone']
        password = request.form['password']
        confirm = request.form['confirm_password']
        tin = request.form.get('tin', '')
        is_vat = request.form.get('is_vat') == 'on'

        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(phone=phone).first():
            flash('Phone number already registered', 'danger')
            return redirect(url_for('register'))

        user = User(shop_name=shop_name, phone=phone, tin=tin, is_vat_registered=is_vat)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']
        remember = 'remember' in request.form
        user = User.query.filter_by(phone=phone).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.shop_name}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid phone number or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ------------------- DASHBOARD -------------------
@app.route('/dashboard')
@login_required
def dashboard():
    today = date.today()
    sales_today = Sale.query.filter_by(user_id=current_user.id).filter(db.func.date(Sale.created_at) == today).all()
    total_sales_today = sum(s.total_amount for s in sales_today)
    total_profit_today = sum(s.profit for s in sales_today)

    products = Product.query.filter_by(user_id=current_user.id).all()
    total_stock_value_cost = sum(p.cost_price * p.quantity for p in products)
    total_stock_value_price = sum(p.selling_price * p.quantity for p in products)
    low_stock_count = sum(1 for p in products if p.quantity <= p.low_stock_threshold)

    tax_due_date = None
    days_left = None
    if current_user.tin and current_user.is_vat_registered:
        today_dt = datetime.today()
        if today_dt.day < 20:
            tax_due_date = datetime(today_dt.year, today_dt.month, 20)
        else:
            next_month = today_dt.replace(day=1) + timedelta(days=32)
            tax_due_date = datetime(next_month.year, next_month.month, 20)
        days_left = (tax_due_date - datetime.today()).days

    return render_template('dashboard.html',
                         total_sales_today=total_sales_today,
                         total_profit_today=total_profit_today,
                         total_stock_value_cost=total_stock_value_cost,
                         total_stock_value_price=total_stock_value_price,
                         low_stock_count=low_stock_count,
                         tax_due_date=tax_due_date,
                         days_left=days_left)

# ------------------- PRODUCT MANAGEMENT -------------------
@app.route('/products')
@login_required
def products():
    products = Product.query.filter_by(user_id=current_user.id).order_by(Product.name).all()
    return render_template('products.html', products=products)

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        name = request.form['name']
        cost_price = float(request.form['cost_price'])
        selling_price = float(request.form['selling_price'])
        quantity = int(request.form['quantity'])
        threshold = int(request.form.get('low_stock_threshold', 5))

        product = Product(user_id=current_user.id, name=name, cost_price=cost_price,
                         selling_price=selling_price, quantity=quantity, low_stock_threshold=threshold)
        db.session.add(product)
        db.session.commit()

        if quantity > 0:
            hist = StockHistory(user_id=current_user.id, product_id=product.id,
                               change_type='add', quantity=quantity, note='Initial stock')
            db.session.add(hist)
            db.session.commit()

        flash('Product added successfully', 'success')
        return redirect(url_for('products'))
    return render_template('add_product.html')

@app.route('/edit_product/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_product(id):
    product = Product.query.get_or_404(id)
    if product.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('products'))

    if request.method == 'POST':
        product.name = request.form['name']
        product.cost_price = float(request.form['cost_price'])
        product.selling_price = float(request.form['selling_price'])
        product.low_stock_threshold = int(request.form.get('low_stock_threshold', 5))
        db.session.commit()
        flash('Product updated', 'success')
        return redirect(url_for('products'))
    return render_template('edit_product.html', product=product)

@app.route('/delete_product/<int:id>')
@login_required
def delete_product(id):
    product = Product.query.get_or_404(id)
    if product.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('products'))
    sales = Sale.query.filter_by(product_id=id).first()
    if sales:
        flash('Cannot delete product with sales history', 'danger')
    else:
        db.session.delete(product)
        db.session.commit()
        flash('Product deleted', 'success')
    return redirect(url_for('products'))

@app.route('/adjust_stock/<int:id>', methods=['POST'])
@login_required
def adjust_stock(id):
    product = Product.query.get_or_404(id)
    if product.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('products'))
    qty = int(request.form['quantity'])
    note = request.form.get('note', '')
    product.quantity += qty
    hist = StockHistory(user_id=current_user.id, product_id=product.id,
                       change_type='add' if qty > 0 else 'remove',
                       quantity=abs(qty), note=note)
    db.session.add(hist)
    db.session.commit()
    flash(f'Stock adjusted by {qty}', 'success')
    return redirect(url_for('products'))

# ------------------- SELLING -------------------
@app.route('/sell', methods=['GET', 'POST'])
@login_required
def sell():
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        quantity = int(request.form['quantity'])
        product = Product.query.get_or_404(product_id)
        if product.user_id != current_user.id:
            flash('Invalid product', 'danger')
            return redirect(url_for('sell'))
        if product.quantity < quantity:
            flash(f'Not enough stock. Only {product.quantity} available', 'danger')
            return redirect(url_for('sell'))

        total = product.selling_price * quantity
        profit = (product.selling_price - product.cost_price) * quantity

        sale = Sale(user_id=current_user.id, product_id=product.id, quantity=quantity,
                   selling_price_at_time=product.selling_price, cost_price_at_time=product.cost_price,
                   total_amount=total, profit=profit)
        product.quantity -= quantity
        db.session.add(sale)
        db.session.commit()

        hist = StockHistory(user_id=current_user.id, product_id=product.id,
                           change_type='remove', quantity=quantity, note=f'Sold {quantity} units')
        db.session.add(hist)
        db.session.commit()

        flash(f'Sold {quantity} x {product.name} for TZS {total:,.0f}', 'success')
        return redirect(url_for('sell'))

    products = Product.query.filter_by(user_id=current_user.id).filter(Product.quantity > 0).order_by(Product.name).all()
    return render_template('sell.html', products=products)

# ------------------- SALES REPORT -------------------
@app.route('/sales_report')
@login_required
def sales_report():
    filter_type = request.args.get('filter', 'today')
    start_date = None
    end_date = None
    today_date = date.today()

    try:
        if filter_type == 'today':
            start_date = today_date
            end_date = today_date
        elif filter_type == 'week':
            start_date = today_date - timedelta(days=today_date.weekday())
            end_date = today_date
        elif filter_type == 'month':
            start_date = today_date.replace(day=1)
            end_date = today_date
        elif filter_type == 'custom':
            start_str = request.args.get('start')
            end_str = request.args.get('end')
            if start_str and end_str:
                start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
            else:
                flash('Please provide both start and end dates', 'warning')
                return redirect(url_for('sales_report'))
        else:
            start_date = today_date
            end_date = today_date
    except Exception as e:
        flash(f'Invalid date format: {e}', 'danger')
        return redirect(url_for('sales_report'))

    if start_date and end_date:
        sales = Sale.query.filter_by(user_id=current_user.id).filter(
            db.func.date(Sale.created_at) >= start_date,
            db.func.date(Sale.created_at) <= end_date
        ).order_by(Sale.created_at.desc()).all()
        total_amount = sum(s.total_amount for s in sales)
        total_profit = sum(s.profit for s in sales)
    else:
        sales = Sale.query.filter_by(user_id=current_user.id).order_by(Sale.created_at.desc()).limit(100).all()
        total_amount = sum(s.total_amount for s in sales)
        total_profit = sum(s.profit for s in sales)

    return render_template('sales_report.html',
                         sales=sales,
                         total_amount=total_amount,
                         total_profit=total_profit,
                         filter_type=filter_type)

# ------------------- LOW STOCK -------------------
@app.route('/low_stock')
@login_required
def low_stock():
    products = Product.query.filter_by(user_id=current_user.id).filter(Product.quantity <= Product.low_stock_threshold).all()
    return render_template('low_stock.html', products=products)

# ------------------- TAX REMINDER -------------------
@app.route('/tax_reminder')
@login_required
def tax_reminder():
    user = current_user
    vat_due = 0
    if user.is_vat_registered and user.tin:
        current_month_start = datetime(datetime.today().year, datetime.today().month, 1)
        sales_this_month = Sale.query.filter_by(user_id=user.id).filter(Sale.created_at >= current_month_start).all()
        total_sales = sum(s.total_amount for s in sales_this_month)
        vat_due = total_sales * 0.18
    due_date = None
    today_dt = datetime.today()
    if today_dt.day < 20:
        due_date = datetime(today_dt.year, today_dt.month, 20)
    else:
        next_month = today_dt.replace(day=1) + timedelta(days=32)
        due_date = datetime(next_month.year, next_month.month, 20)
    days_left = (due_date - today_dt).days
    return render_template('tax_reminder.html', vat_due=vat_due, due_date=due_date, days_left=days_left, tin=user.tin)

# ------------------- PROFILE -------------------
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.shop_name = request.form['shop_name']
        current_user.tin = request.form.get('tin', '')
        current_user.is_vat_registered = 'is_vat' in request.form
        db.session.commit()
        flash('Profile updated', 'success')
        return redirect(url_for('dashboard'))
    return render_template('profile.html', user=current_user)

# ------------------- ADMIN RESET COUNTER (optional) -------------------
@app.route('/admin/reset_counter')
@login_required
@admin_required
def reset_counter():
    stat = SiteStat.query.filter_by(name='total_visitors').first()
    if stat:
        stat.value = 0
        db.session.commit()
        flash('Visitor counter has been reset to 0.', 'success')
    else:
        flash('Counter not found.', 'danger')
    return redirect(url_for('dashboard'))

# ------------------- PWA -------------------
@app.route('/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')

@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js', mimetype='application/javascript')

# ------------------- INIT DB AND ADMIN -------------------
def init_db():
    db.create_all()
    # Create admin user if not exists (first user becomes admin)
    if not User.query.first():
        admin = User(shop_name='Admin Shop', phone='0712345678', tin='123456789', is_vat_registered=True, is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created: phone 0712345678, password admin123")
    # Ensure counter exists
    if not SiteStat.query.filter_by(name='total_visitors').first():
        counter = SiteStat(name='total_visitors', value=0)
        db.session.add(counter)
        db.session.commit()

with app.app_context():
    init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
