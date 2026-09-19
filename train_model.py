import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 10 Categories as defined in Phase 1 requirements:
CATEGORIES = [
    "Food",
    "Travel",
    "Transportation",
    "Shopping",
    "Utilities",
    "Bills",
    "Healthcare",
    "Education",
    "Entertainment",
    "Other"
]

# Comprehensive dataset with realistic merchant names, items, receipts, and descriptions
DATASET = [
    # FOOD
    ("McDonald's burger french fries coke meal combo", "Food"),
    ("Starbucks iced latte cappuccino caramel macchiato coffee", "Food"),
    ("Dominos pizza garlic bread cheese dip dinner", "Food"),
    ("Subway roasted chicken sub sandwich cookie drink", "Food"),
    ("KFC fried chicken bucket zinger burger pepsi", "Food"),
    ("Pizza Hut pan pizza pasta garlic bread", "Food"),
    ("Burger King whopper meal onion rings shake", "Food"),
    ("Haldirams thali chole bhature sweets snacks", "Food"),
    ("Barbeque Nation buffet dinner dining grill", "Food"),
    ("Swiggy food delivery order biryani curd raita", "Food"),
    ("Zomato restaurant food order noodles fried rice", "Food"),
    ("Chai Point tea samosa breakfast ginger chai", "Food"),
    ("Cafe Coffee Day espresso muffin mocha brownie", "Food"),
    ("Grocery store milk bread eggs butter curd", "Food"),
    ("Supermarket vegetables fruits apples bananas tomatoes potatoes", "Food"),
    ("Local bakery cake pastries cookies sourdough bread", "Food"),
    ("Dunkin Donuts glazed donuts iced coffee breakfast", "Food"),
    ("Taco Bell tacos burrito nachos cheese sauce", "Food"),
    ("Chipotle chicken bowl guacamole chips salsa", "Food"),
    ("Baskin Robbins ice cream waffle cone scoop sundae", "Food"),
    ("Restaurant dinner seafood grilled fish salad wine", "Food"),
    ("Dhaba punjabi dal makhani paneer butter masala roti", "Food"),
    ("Blinkit grocery vegetables onions milk cheese bread", "Food"),
    ("Zepto quick grocery fruits curd oil rice spices", "Food"),
    ("Instamart snacks biscuits cold drinks chips chocolates", "Food"),

    # TRAVEL
    ("MakeMyTrip flight booking Mumbai to Delhi indigo", "Food" if False else "Travel"),
    ("Goibibo hotel room booking 2 nights stay resort", "Travel"),
    ("Agoda hotel reservation deluxe room breakfast included", "Travel"),
    ("Airbnb apartment rental weekend stay host", "Travel"),
    ("Cleartrip round trip flight ticket air india", "Travel"),
    ("Booking.com hotel luxury suite check in check out", "Travel"),
    ("IRCTC train ticket booking sleeper class 3A rajdhani", "Travel"),
    ("Expedia international flight tickets to Dubai", "Travel"),
    ("Trivago budget hotel room stay accommodation", "Travel"),
    ("Marriott hotel stay executive room dinner", "Travel"),
    ("Taj Hotel luxury accommodation dining spa", "Travel"),
    ("Travel insurance policy trip cancellation coverage", "Travel"),
    ("Passport visa application fee embassy processing", "Travel"),
    ("Tour package Goa 3 days 4 nights sightseeing", "Travel"),
    ("Airport baggage excess luggage fee airline check-in", "Travel"),
    ("Hostel world backpacker bunk bed booking", "Travel"),
    ("Cruise liner holiday vacation booking ocean cabin", "Travel"),
    ("Foreign exchange currency conversion travel card", "Travel"),

    # TRANSPORTATION
    ("Uber cab ride trip fare downtown city center", "Transportation"),
    ("Ola cabs auto ride driver fare surge", "Transportation"),
    ("Rapido bike taxi ride fare commute", "Transportation"),
    ("Metro rail smart card recharge metro tokens", "Transportation"),
    ("Indian Oil petrol pump fuel gasoline diesel", "Transportation"),
    ("Bharat Petroleum BPCL petrol refill car tank", "Transportation"),
    ("Hindustan Petroleum HPCL diesel refill", "Transportation"),
    ("Shell petrol station v-power fuel engine oil", "Transportation"),
    ("Fastag toll plaza payment national highway toll", "Transportation"),
    ("City parking fee hourly valet parking ticket", "Transportation"),
    ("Car service center oil change wheel alignment tyre rotation", "Transportation"),
    ("Bike maintenance brake pads chain lube servicing", "Transportation"),
    ("Bus ticket interstate travel sleeper coach ticket", "Transportation"),
    ("Car wash detailing wax interior vacuum polish", "Transportation"),
    ("EV electric vehicle charging station recharge kw/h", "Transportation"),
    ("Railway local train monthly season ticket pass", "Transportation"),
    ("Auto rickshaw meter fare local ride", "Transportation"),

    # SHOPPING
    ("Amazon online shopping order wireless earbuds electronics", "Shopping"),
    ("Flipkart shopping order sneakers t-shirt clothing", "Shopping"),
    ("Myntra casual shirts jeans jacket fashion apparel", "Shopping"),
    ("Zara cotton shirt trousers blazer casual wear", "Shopping"),
    ("H&M hoodie denim jacket t-shirt socks", "Shopping"),
    ("Nike running shoes sports apparel gym wear", "Shopping"),
    ("Adidas sneakers athletic track pants sweatshirt", "Shopping"),
    ("IKEA furniture study desk office chair lamp bookshelf", "Shopping"),
    ("Croma electronics laptop sleeve hdmi cable mouse", "Shopping"),
    ("Reliance Digital bluetooth speaker power bank earphones", "Shopping"),
    ("Decathlon sports gear tent badminton racket shoes", "Shopping"),
    ("Sephora cosmetics perfume foundation lipstick skincare", "Shopping"),
    ("Nykaa beauty face wash sunscreen serum moisturizer", "Shopping"),
    ("Uniqlo heattech jacket merino wool sweater jeans", "Shopping"),
    ("Westside ladies dress casual tops footwear", "Shopping"),
    ("Pantaloons ethnic wear kurta pajama formal shirt", "Shopping"),
    ("Apple Store iphone case screen protector charger adapter", "Shopping"),

    # UTILITIES
    ("Electricity bill payment power supply kilowatt hours", "Utilities"),
    ("Water supply department quarterly water utility bill", "Utilities"),
    ("Piped natural gas PNG cooking gas bill adani gas", "Utilities"),
    ("LPG cooking gas cylinder refill indane hp gas bharat gas", "Utilities"),
    ("Municipal corporation property tax sewage drainage charges", "Utilities"),
    ("Garbage collection and disposal monthly utility fee", "Utilities"),
    ("Society maintenance charges residential maintenance lift security", "Utilities"),
    ("Solar power grid maintenance monthly fee", "Utilities"),
    ("Water purifier RO service candle filter replacement", "Utilities"),
    ("Home electricity connection surcharge bill state electricity board", "Utilities"),

    # BILLS
    ("Airtel postpaid mobile bill recharge data voice plan", "Bills"),
    ("Jio prepaid recharge plan unlimited 5g 84 days", "Bills"),
    ("Vodafone Idea VI monthly postpaid cellular bill", "Bills"),
    ("ACT Fibernet broadband wifi fiber optic monthly internet bill", "Bills"),
    ("Tata Play DTH set top box subscription tv channels", "Bills"),
    ("Credit card bill payment monthly statement minimum due", "Bills"),
    ("Bank home loan EMI deduction housing finance", "Bills"),
    ("Personal loan EMI monthly installment bank debit", "Bills"),
    ("Car vehicle loan monthly installment EMI payment", "Bills"),
    ("House rent monthly landlord tenant payment", "Bills"),
    ("Locker rent annual bank safe deposit box", "Bills"),

    # HEALTHCARE
    ("Apollo Pharmacy medicines prescription pills syrup tablets", "Healthcare"),
    ("Medplus pharmacy paracetamol antibiotics vitamins painkiller", "Healthcare"),
    ("Dr Lal PathLabs blood test health checkup full body profile", "Healthcare"),
    ("Dental clinic teeth cleaning root canal cavity filling", "Healthcare"),
    ("Hospital consultation doctor OPD fees physician specialist", "Healthcare"),
    ("1mg online medicine delivery diagnostic test lab", "Healthcare"),
    ("PharmEasy health care supplements protein powder bandages", "Healthcare"),
    ("Optometrist eye exam spectacle lenses frames contact lenses", "Healthcare"),
    ("Health insurance premium policy mediclaim annual coverage", "Healthcare"),
    ("Physiotherapy session muscle rehabilitation therapy", "Healthcare"),
    ("Vaccination injection clinic immunization pediatric", "Healthcare"),
    ("X-Ray MRI scan CT scan ultrasound diagnostic center", "Healthcare"),
    ("Dermatologist skin treatment acne consultation ointment", "Healthcare"),

    # EDUCATION
    ("University semester tuition fee college admission examination fee", "Education"),
    ("School term fees kindergarten books uniform activity fee", "Education"),
    ("Coursera online specialization certification course fee", "Education"),
    ("Udemy programming web development python course purchase", "Education"),
    ("Bookstore textbooks engineering mathematics physics novel", "Education"),
    ("Stationery notebooks pens pencils markers paper binder", "Education"),
    ("EdX professional certificate machine learning data science", "Education"),
    ("Coaching classes tuition institute entrance exam preparation", "Education"),
    ("College library fine annual membership research journals", "Education"),
    ("Masterclass annual subscription online learning tutorials", "Education"),
    ("Duolingo language learning subscription super duolingo", "Education"),

    # ENTERTAINMENT
    ("PVR Cinemas movie tickets pop corn coke multiplex IMAX", "Entertainment"),
    ("INOX movies tickets recliner screen show evening", "Entertainment"),
    ("BookMyShow cinema concert event tickets booking", "Entertainment"),
    ("Netflix monthly streaming subscription 4K UHD plan", "Entertainment"),
    ("Spotify premium individual music streaming subscription", "Entertainment"),
    ("Disney+ Hotstar annual VIP subscription sports live", "Entertainment"),
    ("Amazon Prime Video membership streaming subscription", "Entertainment"),
    ("YouTube Premium ad-free music background play", "Entertainment"),
    ("Gaming Playstation Plus Xbox Game Pass steam games purchase", "Entertainment"),
    ("Amusement park theme park entry tickets roller coaster rides", "Entertainment"),
    ("Bowling alley arcade games laser tag fun zone", "Entertainment"),
    ("Live concert music festival standup comedy show ticket", "Entertainment"),
    ("Museum art gallery exhibition admission pass", "Entertainment"),

    # OTHER
    ("Charity donation relief fund NGO contribution", "Other"),
    ("Legal lawyer consultation legal notice drafting notary", "Other"),
    ("Pet shop dog food cat litter veterinary checkup pet toys", "Other"),
    ("Laundry dry cleaning suit wash iron services", "Other"),
    ("Courier speed post shipping parcel delivery fee", "Other"),
    ("Government service fee stamp paper affidavit registration", "Other"),
    ("Key maker locksmith lock replacement duplicate keys", "Other"),
    ("Miscellaneous unexpected expense cash withdrawal", "Other"),
    ("Religious offering temple church donation flowers", "Other")
]

# Data augmentation: expand vocabulary and variations to ensure robust generalized classification
def augment_data(dataset):
    augmented = []
    prefixes = ["Payment for ", "Invoice for ", "Bill for ", "Receipt: ", "Purchased ", "Order: ", "Store: "]
    suffixes = [" with cash", " via UPI", " credit card", " bill", " store receipt", " online payment", " transaction"]
    
    for text, cat in dataset:
        augmented.append((text, cat))
        augmented.append((text.lower(), cat))
        augmented.append((text.upper(), cat))
        # Mix prefixes and suffixes
        for p in prefixes[:3]:
            augmented.append((f"{p}{text}", cat))
        for s in suffixes[:3]:
            augmented.append((f"{text}{s}", cat))
    return augmented

def train():
    print("[AI Service] Augmenting dataset...")
    full_data = augment_data(DATASET)
    print(f"[AI Service] Total training samples: {len(full_data)}")
    
    texts, labels = zip(*full_data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000,
            token_pattern=r'(?u)\b\w+\b'
        )),
        ('clf', LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42
        ))
    ])
    
    print("[AI Service] Fitting pipeline on training set...")
    pipeline.fit(X_train, y_train)
    
    print("[AI Service] Evaluating model on test set...")
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[AI Service] Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    # Save artifacts in the current directory
    out_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(out_dir, "model.joblib")
    
    print(f"[AI Service] Saving pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("[AI Service] Model training and serialization complete.")

if __name__ == "__main__":
    train()
