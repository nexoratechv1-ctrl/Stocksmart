import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    email = db.Column(db.String(100))
    tin = db.Column(db.String(50))
    is_vat_registered = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(100), nullable=False)
    cost_price = db.Column(db.Float, nullable=False)
    selling_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Float, default=0)
    low_stock_threshold = db.Column(db.Float, default=5)
    unit = db.Column(db.String(20), default='piece')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'))
    quantity = db.Column(db.Float, nullable=False)
    selling_price_at_time = db.Column(db.Float)
    cost_price_at_time = db.Column(db.Float)
    total_amount = db.Column(db.Float)
    profit = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StockHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'))
    change_type = db.Column(db.String(20))
    quantity = db.Column(db.Float)
    note = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_approved = db.Column(db.Boolean, default=True)

class AnomalyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.Date, nullable=False)
    anomaly_type = db.Column(db.String(20))
    severity = db.Column(db.Float)
    expected_sales = db.Column(db.Float)
    actual_sales = db.Column(db.Float)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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

# ------------------- ANOMALY DETECTION -------------------
def detect_anomalies(user_id):
    end_date = date.today()
    start_date = end_date - timedelta(30)
    sales = Sale.query.filter_by(user_id=user_id).filter(Sale.created_at >= start_date).all()
    if len(sales) < 7:
        return
    daily = {}
    for s in sales:
        d = s.created_at.date()
        daily[d] = daily.get(d, 0) + s.total_amount
    arr = [daily.get(end_date - timedelta(i), 0) for i in range(30)]
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return
    today_sales = daily.get(end_date, 0)
    z = (today_sales - mean) / std
    if abs(z) > 1.5:
        typ = 'spike' if today_sales > mean else 'drop'
        existing = AnomalyLog.query.filter_by(user_id=user_id, date=end_date).first()
        if not existing:
            log = AnomalyLog(
                user_id=user_id,
                date=end_date,
                anomaly_type=typ,
                severity=abs(z),
                expected_sales=mean,
                actual_sales=today_sales,
                notes=f"{typ} detected. Expected ~{mean:.0f} got {today_sales:.0f}"
            )
            db.session.add(log)
            db.session.commit()

# ------------------- AI FORECAST -------------------
def train_sales_model(user_id):
    sales = Sale.query.filter_by(user_id=user_id).all()
    if len(sales) < 7:
        return None, None
    data = {}
    for s in sales:
        d = s.created_at.strftime('%Y-%m-%d')
        data[d] = data.get(d, 0) + s.total_amount
    df = pd.DataFrame(list(data.items()), columns=['date', 'sales'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['day_num'] = (df['date'] - df['date'].min()).dt.days
    X = df[['day_num']].values
    y = df['sales'].values
    model = LinearRegression()
    model.fit(X, y)
    return model, df['date'].max()

# ------------------- ROUTES -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        shop_name = request.form['shop_name']
        phone = request.form['phone']
        email = request.form.get('email')
        tin = request.form.get('tin', '')
        is_vat = 'is_vat' in request.form
        password = request.form['password']
        confirm = request.form['confirm_password']
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(phone=phone).first():
            flash('Phone number already registered', 'danger')
            return redirect(url_for('register'))
        user = User(
            shop_name=shop_name,
            phone=phone,
            email=email,
            tin=tin,
            is_vat_registered=is_vat
        )
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

@app.route('/dashboard')
@login_required
def dashboard():
    today = date.today()
    sales_today = Sale.query.filter_by(user_id=current_user.id).filter(db.func.date(Sale.created_at) == today).all()
    total_sales_today = sum(s.total_amount for s in sales_today)
    total_profit_today = sum(s.profit for s in sales_today)

    products = Product.query.filter_by(user_id=current_user.id).all()
    total_stock_value_cost = sum(p.cost_price * p.quantity for p in products)
    low_stock_count = sum(1 for p in products if p.quantity <= p.low_stock_threshold)

    tax_due_date = None
    days_left = None
    if current_user.tin and current_user.is_vat_registered:
        td = datetime.today()
        if td.day < 20:
            tax_due_date = datetime(td.year, td.month, 20)
        else:
            next_month = td.replace(day=1) + timedelta(days=32)
            tax_due_date = datetime(next_month.year, next_month.month, 20)
        days_left = (tax_due_date - td).days

    detect_anomalies(current_user.id)
    today_anomaly = AnomalyLog.query.filter_by(user_id=current_user.id, date=today).first()

    return render_template('dashboard.html',
                         total_sales_today=total_sales_today,
                         total_profit_today=total_profit_today,
                         total_stock_value_cost=total_stock_value_cost,
                         low_stock_count=low_stock_count,
                         tax_due_date=tax_due_date,
                         days_left=days_left,
                         today_anomaly=today_anomaly)

@app.route('/products')
@login_required
def products():
    prods = Product.query.filter_by(user_id=current_user.id).order_by(Product.name).all()
    return render_template('products.html', products=prods)

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        name = request.form['name']
        cost_price = float(request.form['cost_price'])
        selling_price = float(request.form['selling_price'])
        quantity = float(request.form['quantity'])
        threshold = float(request.form.get('low_stock_threshold', 5))
        unit = request.form.get('unit', 'piece')
        product = Product(
            user_id=current_user.id,
            name=name,
            cost_price=cost_price,
            selling_price=selling_price,
            quantity=quantity,
            low_stock_threshold=threshold,
            unit=unit
        )
        db.session.add(product)
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
        product.low_stock_threshold = float(request.form.get('low_stock_threshold', 5))
        product.unit = request.form.get('unit', 'piece')
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
    if Sale.query.filter_by(product_id=id).first():
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
    qty = float(request.form['quantity'])
    note = request.form.get('note', '')
    product.quantity += qty
    hist = StockHistory(
        user_id=current_user.id,
        product_id=product.id,
        change_type='add' if qty > 0 else 'remove',
        quantity=abs(qty),
        note=note
    )
    db.session.add(hist)
    db.session.commit()
    flash(f'Stock adjusted by {qty}', 'success')
    return redirect(url_for('products'))

@app.route('/sell', methods=['GET', 'POST'])
@login_required
def sell():
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        quantity = float(request.form['quantity'])
        product = Product.query.get_or_404(product_id)
        if product.user_id != current_user.id:
            flash('Invalid product', 'danger')
            return redirect(url_for('sell'))
        if product.quantity < quantity:
            flash(f'Not enough stock. Only {product.quantity} {product.unit} available', 'danger')
            return redirect(url_for('sell'))
        total = product.selling_price * quantity
        profit = (product.selling_price - product.cost_price) * quantity
        sale = Sale(
            user_id=current_user.id,
            product_id=product.id,
            quantity=quantity,
            selling_price_at_time=product.selling_price,
            cost_price_at_time=product.cost_price,
            total_amount=total,
            profit=profit
        )
        product.quantity -= quantity
        db.session.add(sale)
        db.session.commit()
        flash(f'Sold {quantity} {product.unit} of {product.name} for TZS {total:,.0f}', 'success')
        return redirect(url_for('sell'))
    products = Product.query.filter_by(user_id=current_user.id).filter(Product.quantity > 0).all()
    return render_template('sell.html', products=products)

@app.route('/sales_report')
@login_required
def sales_report():
    filter_type = request.args.get('filter', 'today')
    today_date = date.today()
    if filter_type == 'today':
        start = end = today_date
    elif filter_type == 'week':
        start = today_date - timedelta(days=today_date.weekday())
        end = today_date
    elif filter_type == 'month':
        start = today_date.replace(day=1)
        end = today_date
    elif filter_type == 'custom':
        start_str = request.args.get('start')
        end_str = request.args.get('end')
        start = datetime.strptime(start_str, '%Y-%m-%d').date() if start_str else today_date
        end = datetime.strptime(end_str, '%Y-%m-%d').date() if end_str else today_date
    else:
        start = end = today_date
    sales = Sale.query.filter_by(user_id=current_user.id).filter(
        db.func.date(Sale.created_at) >= start,
        db.func.date(Sale.created_at) <= end
    ).order_by(Sale.created_at.desc()).all()
    total_amount = sum(s.total_amount for s in sales)
    total_profit = sum(s.profit for s in sales)
    return render_template('sales_report.html', sales=sales, total_amount=total_amount, total_profit=total_profit, filter_type=filter_type)

@app.route('/low_stock')
@login_required
def low_stock():
    products = Product.query.filter_by(user_id=current_user.id).filter(Product.quantity <= Product.low_stock_threshold).all()
    return render_template('low_stock.html', products=products)

@app.route('/tax_reminder')
@login_required
def tax_reminder():
    user = current_user
    vat_due = 0
    if user.is_vat_registered and user.tin:
        month_start = datetime(datetime.today().year, datetime.today().month, 1)
        sales_this_month = Sale.query.filter_by(user_id=user.id).filter(Sale.created_at >= month_start).all()
        total_sales = sum(s.total_amount for s in sales_this_month)
        vat_due = total_sales * 0.18
    td = datetime.today()
    if td.day < 20:
        due_date = datetime(td.year, td.month, 20)
    else:
        next_month = td.replace(day=1) + timedelta(days=32)
        due_date = datetime(next_month.year, next_month.month, 20)
    days_left = (due_date - td).days
    return render_template('tax_reminder.html', vat_due=vat_due, due_date=due_date, days_left=days_left, tin=user.tin)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.shop_name = request.form['shop_name']
        current_user.email = request.form.get('email', '')
        current_user.tin = request.form.get('tin', '')
        current_user.is_vat_registered = 'is_vat' in request.form
        db.session.commit()
        flash('Profile updated', 'success')
        return redirect(url_for('dashboard'))
    return render_template('profile.html', user=current_user)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        if not current_user.is_authenticated:
            flash('Please login to leave feedback.', 'warning')
            return redirect(url_for('login'))
        rating = int(request.form.get('rating', 0))
        comment = request.form.get('comment', '').strip()
        if rating < 1 or rating > 5 or not comment:
            flash('Please provide a valid rating (1-5) and comment.', 'danger')
            return redirect(url_for('feedback'))
        new_comment = Comment(user_id=current_user.id, rating=rating, comment=comment)
        db.session.add(new_comment)
        db.session.commit()
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('feedback'))
    comments = Comment.query.filter_by(is_approved=True).order_by(Comment.created_at.desc()).all()
    return render_template('feedback.html', comments=comments)

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_sales_all = Sale.query.count()
    total_revenue_all = db.session.query(db.func.sum(Sale.total_amount)).scalar() or 0
    users = User.query.order_by(User.created_at.desc()).all()
    comments = Comment.query.order_by(Comment.created_at.desc()).all()
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_sales_all=total_sales_all,
                         total_revenue_all=total_revenue_all,
                         users=users,
                         comments=comments)

@app.route('/ai-forecast')
@login_required
def ai_forecast():
    model, last_date = train_sales_model(current_user.id)
    if model is None:
        flash('Not enough sales data (need at least 7 days).', 'warning')
        return redirect(url_for('dashboard'))
    last_day_num = (last_date - datetime.now().date()).days + 30
    future_days = np.array([[last_day_num + i] for i in range(1, 8)])
    predictions = model.predict(future_days)
    predictions = [max(0, round(p)) for p in predictions]
    forecast = [{'day': i+1, 'sales': p} for i, p in enumerate(predictions)]
    advice = []
    avg_pred = sum(predictions) / len(predictions)
    advice.append(f"📈 Predicted average daily sales next week: TZS {avg_pred:,.0f}")
    if predictions[-1] > predictions[0] * 1.2:
        advice.append("⚠️ Sales are increasing! Order extra stock for next week.")
    elif predictions[-1] < predictions[0] * 0.8:
        advice.append("ℹ️ Sales may decrease. Avoid overstocking.")
    else:
        advice.append("✅ Sales are stable. Maintain current stock levels.")
    return render_template('ai_forecast.html', forecast=forecast, advice=advice)

@app.route('/anomaly_history')
@login_required
def anomaly_history():
    anomalies = AnomalyLog.query.filter_by(user_id=current_user.id).order_by(AnomalyLog.date.desc()).all()
    return render_template('anomaly_history.html', anomalies=anomalies)

@app.route('/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')

@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js', mimetype='application/javascript')

# ------------------- INIT -------------------
def init_db():
    db.create_all()
    if not User.query.first():
        admin = User(
            shop_name='Admin Shop',
            phone='0712345678',
            email='admin@stocksmart.com',
            tin='123456789',
            is_vat_registered=True,
            is_admin=True
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created: phone 0712345678, password admin123")

with app.app_context():
    init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True).filter_by(user_id=current_user.id).filter(Product.quantity>0).all()
    return render_template('sell.html', products=products)

@app.route('/sales_report')
@login_required
def sales_report():
    filter_type = request.args.get('filter','today')
    today_date = date.today()
    if filter_type == 'today': start=end=today_date
    elif filter_type == 'week': start = today_date - timedelta(days=today_date.weekday()); end=today_date
    elif filter_type == 'month': start = today_date.replace(day=1); end=today_date
    elif filter_type == 'custom':
        start = datetime.strptime(request.args.get('start'),'%Y-%m-%d').date() if request.args.get('start') else today_date
        end = datetime.strptime(request.args.get('end'),'%Y-%m-%d').date() if request.args.get('end') else today_date
    else: start=end=today_date
    sales = Sale.query.filter_by(user_id=current_user.id).filter(db.func.date(Sale.created_at)>=start, db.func.date(Sale.created_at)<=end).order_by(Sale.created_at.desc()).all()
    total_amount = sum(s.total_amount for s in sales); total_profit = sum(s.profit for s in sales)
    return render_template('sales_report.html', sales=sales, total_amount=total_amount, total_profit=total_profit, filter_type=filter_type)

@app.route('/low_stock')
@login_required
def low_stock():
    return render_template('low_stock.html', products=Product.query.filter_by(user_id=current_user.id).filter(Product.quantity <= Product.low_stock_threshold).all())

@app.route('/tax_reminder')
@login_required
def tax_reminder():
    user = current_user
    vat_due = 0
    if user.is_vat_registered and user.tin:
        month_start = datetime(datetime.today().year, datetime.today().month, 1)
        sales = Sale.query.filter_by(user_id=user.id).filter(Sale.created_at >= month_start).all()
        total_sales = sum(s.total_amount for s in sales)
        vat_due = total_sales * 0.18
    td = datetime.today()
    due_date = datetime(td.year, td.month, 20) if td.day < 20 else datetime((td.replace(day=1)+timedelta(32)).year, (td.replace(day=1)+timedelta(32)).month, 20)
    days_left = (due_date - td).days
    return render_template('tax_reminder.html', vat_due=vat_due, due_date=due_date, days_left=days_left, tin=user.tin)

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.shop_name = request.form['shop_name']
        current_user.email = request.form.get('email','')
        current_user.tin = request.form.get('tin','')
        current_user.is_vat_registered = 'is_vat' in request.form
        db.session.commit(); flash('Profile updated','success'); return redirect(url_for('dashboard'))
    return render_template('profile.html', user=current_user)

@app.route('/feedback', methods=['GET','POST'])
def feedback():
    if request.method == 'POST':
        if not current_user.is_authenticated: flash('Login to comment','warning'); return redirect(url_for('login'))
        rating = int(request.form.get('rating',0)); comment = request.form.get('comment','').strip()
        if rating<1 or rating>5 or not comment: flash('Valid rating and comment required','danger')
        else:
            db.session.add(Comment(user_id=current_user.id, rating=rating, comment=comment)); db.session.commit()
            flash('Thank you for feedback!','success')
        return redirect(url_for('feedback'))
    comments = Comment.query.filter_by(is_approved=True).order_by(Comment.created_at.desc()).all()
    return render_template('feedback.html', comments=comments)

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', total_users=User.query.count(),
                           total_sales_all=Sale.query.count(),
                           total_revenue_all=db.session.query(db.func.sum(Sale.total_amount)).scalar() or 0,
                           users=User.query.order_by(User.created_at.desc()).all(),
                           comments=Comment.query.order_by(Comment.created_at.desc()).all())

@app.route('/ai-forecast')
@login_required
def ai_forecast():
    model, last_date = train_model(current_user.id)
    if model is None: flash('Need at least 7 days of sales data','warning'); return redirect(url_for('dashboard'))
    last_day = (last_date - datetime.now().date()).days + 30
    preds = model.predict(np.array([[last_day + i] for i in range(1,8)]))
    preds = [max(0,round(p)) for p in preds]
    forecast = [{'day':i+1,'sales':p} for i,p in enumerate(preds)]
    advice = [f"📈 Avg predicted daily sales: TZS {sum(preds)/7:,.0f}"]
    if preds[-1] > preds[0]*1.2: advice.append("⚠️ Sales increasing – order extra stock")
    elif preds[-1] < preds[0]*0.8: advice.append("ℹ️ Sales may decrease – avoid overstocking")
    else: advice.append("✅ Sales stable – maintain current levels")
    return render_template('ai_forecast.html', forecast=forecast, advice=advice)

@app.route('/anomaly_history')
@login_required
def anomaly_history():
    return render_template('anomaly_history.html', anomalies=AnomalyLog.query.filter_by(user_id=current_user.id).order_by(AnomalyLog.date.desc()).all())

@app.route('/manifest.json')
def manifest(): return app.send_static_file('manifest.json')
@app.route('/service-worker.js')
def sw(): return app.send_static_file('service-worker.js', mimetype='application/javascript')

# ------------------- INIT -------------------
def init_db():
    db.create_all()
    if not User.query.first():
        admin = User(shop_name='Admin Shop', phone='0712345678', email='admin@stocksmart.com', tin='123456789', is_vat_registered=True, is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin); db.session.commit()
        print("Admin: 0712345678 / admin123")

with app.app_context(): init_db()
if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=True)
