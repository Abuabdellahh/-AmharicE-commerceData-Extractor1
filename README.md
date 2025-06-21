# Amharic E-commerce Data Extractor

Building an Amharic E-commerce Data Extractor for EthioMart

## Project Overview

This project implements a system for extracting business entities (products, prices, and locations) from Amharic e-commerce data collected from Telegram channels. The system includes data collection, preprocessing, NER model training, and vendor analytics components.

## Project Structure

```
amharic-ecommerce-extractor/
├── config/              # Configuration files
│   └── config.yaml      # Project configuration
├── src/                 # Source code
│   ├── data/            # Data processing modules
│   │   ├── telegram_scraper.py
│   │   └── text_preprocessor.py
│   ├── labeling/        # NER labeling modules
│   │   └── conll_labeler.py
│   ├── models/          # Model implementation modules
│   │   ├── ner_model.py
│   │   ├── model_comparison.py
│   │   └── model_interpretability.py
│   └── analytics/       # Vendor analytics modules
│       └── vendor_analytics.py
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore file
├── LICENSE              # Project license
├── Makefile             # Common tasks
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd amharic-ecommerce-extractor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Telegram API credentials
```

## Usage

Use Makefile commands for common tasks:

```bash
# Install dependencies
make install

# Run data collection
make collect_data

# Run preprocessing
make preprocess

# Run NER labeling
make label

# Train model
make train

# Compare models
make compare

# Run vendor analytics
make analytics

# Clean up
make clean
```

## Project Components

### Data Collection
- Telegram scraper for collecting e-commerce data
- Supports multiple channels simultaneously
- Collects metadata (views, forwards, timestamps)

### Data Processing
- Amharic text normalization
- Entity extraction patterns
- Tokenization support
- Unicode normalization

### NER System
- XLM-Roberta fine-tuning
- Model comparison framework
- SHAP and LIME interpretability
- Performance metrics calculation

### Vendor Analytics
- Vendor performance metrics
- Lending score calculation
- Top performing post analysis
- Price point analysis

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SHAP
- LIME

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to EthioMart for the project opportunity
- Special thanks to the 10 Academy program for their support