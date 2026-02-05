import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Professional dark palette
BG_COLOR = "#0D1117"
BOX_COLOR = "#161B22"
EDGE_COLOR = "#30363D"
TEXT_COLOR = "#C9D1D9"
ARROW_COLOR = "#58A6FF"
CNN_COLOR = "#D29922"   # Gold/Amber
RNN_COLOR = "#238636"   # Green
TRANS_COLOR = "#F85149" # Red

def draw_crnn_architecture():
    os.makedirs('assets', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 14), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Helper function for boxes
    def draw_box(x, y, w, h, label, color, sublabel=""):
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.3",
            linewidth=2, edgecolor=EDGE_COLOR, facecolor=BOX_COLOR
        )
        ax.add_patch(rect)
        ax.text(x, y + (1 if sublabel else 0), label, 
                color=color, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        if sublabel:
            ax.text(x, y - 2, sublabel, 
                    color=TEXT_COLOR, ha='center', va='center', 
                    fontsize=9, style='italic')

    # Draw Layers from Bottom to Top
    
    # 1. Input
    draw_box(50, 10, 30, 8, "Input Image", TEXT_COLOR, "(Grayscale 32x128)")
    
    # 2. CNN
    draw_box(50, 30, 40, 12, "Convolutional Layers", CNN_COLOR, "5x Conv + BatchNorm + MaxPool")
    ax.text(80, 30, "Feature Extraction", color=TEXT_COLOR, fontsize=10)

    # 3. Maps to sequence
    draw_box(50, 50, 35, 8, "Feature Sequence", TEXT_COLOR, "(Time-distributed maps)")

    # 4. RNN
    draw_box(50, 70, 40, 12, "Deep Bidirectional LSTM", RNN_COLOR, "2 Layers, 256 Hidden Units")
    ax.text(80, 70, "Recurrent Layers", color=TEXT_COLOR, fontsize=10)

    # 5. Transcription
    draw_box(50, 90, 30, 10, "Transcription Layer", TRANS_COLOR, "Softmax + CTC Decoding")
    ax.text(80, 90, "Final Prediction", color=TEXT_COLOR, fontsize=10)

    # Arrows
    arrow_data = [
        (50, 15, 50, 23), # Input to CNN
        (50, 37, 50, 45), # CNN to Seq
        (50, 55, 50, 63), # Seq to RNN
        (50, 77, 50, 84), # RNN to Trans
    ]
    
    for x1, y1, x2, y2 in arrow_data:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='simple', color=ARROW_COLOR, lw=1, mutation_scale=20))

    # Predicted Sequence text at the very top
    ax.text(50, 97, "\"HELLO WORLD\"", color="#FFFFFF", ha='center', fontsize=16, fontweight='black', bbox=dict(facecolor=HIGHLIGHT_COLOR if 'HIGHLIGHT_COLOR' in locals() else "#238636", alpha=0.5, boxstyle='round,pad=0.5'))

    plt.title("Detailed CRNN Architecture Overview", 
              fontsize=20, fontweight='bold', color=TEXT_COLOR, pad=30)
    
    plt.savefig('assets/crnn_architecture.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print("Architecture image generated successfully.")

if __name__ == "__main__":
    draw_crnn_architecture()
