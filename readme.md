🎭 Fictional Character Personality Matcher

A machine-learning based personality quiz that matches users with fictional characters from different universes based on personality traits.

The project uses a set of questions mapped to psychological traits and compares user responses with predefined character profiles using distance-based similarity.

✨ Features

- 🌍 Multiple universes:
  - Harry Potter
  - Disney
  - Pixar
  - Disney Princesses 👑
- 🧠 Personality matching using ML concepts
- 🎨 Cute Streamlit GUI with soft pink theme
- 🖼 Background image support with fallback color
- 📊 Trait-based character profiles
- ⚡ Fast and lightweight

📁 Project Structure

Fictional Character │ ├── 📁 app │   └──  gui_app.py          # Streamlit GUI │ ├── 📁 assets │   └── background.jpg     # Background image │ ├── 📁 data │   ├── characters.csv     # Characters + traits + universe + movie │   └── questions.json     # Questions and trait mappings │ ├── 📁 models │   └──  knn_model.pkl      # Trained scaler/model file │ ├── 📁 src │   ├── predict.py         # CLI version (optional) │   ├── similarity.py      # Similarity utilities (optional) │   └── train_model.py     # Model training script │ └── 📄 requirement.txt        # Project dependencies



# install dependencies
pip install -r requirement.txt