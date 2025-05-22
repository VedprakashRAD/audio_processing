import subprocess
import sys

def install_dependencies():
    print("Installing required language models...")
    
    # Install spaCy model
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully installed spaCy model")
    except Exception as e:
        print(f"Error installing spaCy model: {e}")
    
    # Install NLTK data
    try:
        import nltk
        nltk.download('vader_lexicon')
        print("Successfully installed NLTK vader_lexicon")
    except Exception as e:
        print(f"Error installing NLTK data: {e}")

if __name__ == "__main__":
    install_dependencies() 