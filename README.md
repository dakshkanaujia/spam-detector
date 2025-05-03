
---

## ⚙️ Setup & Usage

### 1. **Clone the Repository**
```bash
git clone https://github.com/dakshkanaujia/spam-classifier-app.git
cd spam-classifier-app
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Train the Model**
```bash
python train_model.py
```
- This downloads the dataset, preprocesses it, trains the model, and saves it in the `model/` directory.

### 4. **Run the Web App**
```bash
python app.py
```
- Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS (minimal, dashboard-inspired), Chart.js
- **Backend:** Flask (Python)
- **ML:** scikit-learn, pandas, nltk
- **Model:** Multinomial Naive Bayes (can be swapped for Logistic Regression)

---

## 📊 Dataset

- [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Contains 5,574 SMS messages labeled as "ham" (not spam) or "spam".

---

## ✨ Customization

- **Model:**  
  Swap in Logistic Regression or other classifiers in `train_model.py`.
- **Preprocessing:**  
  Tweak the `preprocess` function for your data.
- **UI:**  
  Edit `static/style.css` and `templates/index.html` for your own look.

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Feel free to open an issue or submit a PR.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [scikit-learn](https://scikit-learn.org/)
- [Chart.js](https://www.chartjs.org/)