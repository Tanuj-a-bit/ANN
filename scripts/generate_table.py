import matplotlib.pyplot as plt
import pandas as pd
import os

# Professional dark palette (matching previous graphs)
BG_COLOR = "#0D1117"
HEADER_COLOR = "#1f2937"
ROW_COLOR = "#0D1117"
HIGHLIGHT_COLOR = "#238636" # Green for "Ours"
TEXT_COLOR = "#C9D1D9"
EDGE_COLOR = "#30363D"

def generate_table_image():
    os.makedirs('assets', exist_ok=True)
    
    data = [
        ["[1]", "Traditional OCR", "Template Matching", "Simple implementation", "Poor handwriting support"],
        ["[2]", "HMM-based HWR", "Hidden Markov Models", "Sequential modeling", "Requires segmentation"],
        ["[3]", "CNN-based OCR", "Deep CNN", "Strong feature extraction", "No temporal context"],
        ["[4]", "CNN + RNN", "LSTM", "Sequence modeling", "Unidirectional context"],
        ["[5]", "CRNN + Bi-LSTM", "CNN + Bi-LSTM + CTC", "High accuracy, no segmentation", "Higher training cost"],
        ["[6]", "Transformer HWR", "Self-Attention", "State-of-the-art accuracy", "High latency, heavy compute"],
        ["Ours", "ANN-HWR", "CRNN + Bi-LSTM + CTC", "Real-time, high accuracy", "Limited to Phase-1 scope"]
    ]
    
    columns = ["Ref", "Model / System", "Technique Used", "Key Strength", "Limitation"]
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_COLOR)
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='left')
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        if row == 0: # Header
            cell.set_text_props(weight='bold', color=TEXT_COLOR)
            cell.set_facecolor(HEADER_COLOR)
        else:
            if data[row-1][0] == "Ours": # Highlight "Ours" row
                cell.set_facecolor(HIGHLIGHT_COLOR)
                cell.set_text_props(color="#FFFFFF", weight='bold') # White text for contrast
            else:
                cell.set_facecolor(ROW_COLOR)
                cell.set_text_props(color=TEXT_COLOR)

    plt.title("Literature Survey: Comparative Analysis of HWR Systems", 
              fontsize=18, fontweight='bold', color=TEXT_COLOR, pad=40)
    
    plt.savefig('assets/literature_survey_table.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print("Table image generated successfully in assets/folder.")

if __name__ == "__main__":
    generate_table_image()
