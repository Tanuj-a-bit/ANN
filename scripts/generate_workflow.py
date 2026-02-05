import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Professional dark palette
BG_COLOR = "#0D1117"
BOX_COLOR = "#161B22"
EDGE_COLOR = "#30363D"
TEXT_COLOR = "#C9D1D9"
ARROW_COLOR = "#58A6FF"
HIGHLIGHT_COLOR = "#238636"

def draw_workflow():
    os.makedirs('assets', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Define steps
    steps = [
        "Handwritten Input\n(Image Capture)",
        "Feature Extraction\n(CNN Layers)",
        "Sequence Modeling\n(Bi-LSTM)",
        "Decoding\n(CTC Layer)",
        "Refinement\n(NSR Module)"
    ]
    
    # Box positions and dimensions
    box_width = 18
    box_height = 12
    x_pos = 50
    y_positions = [85, 65, 45, 25, 5]
    
    for i, step in enumerate(steps):
        # Draw Box
        rect = patches.FancyBboxPatch(
            (x_pos - box_width/2, y_positions[i] - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.3",
            linewidth=2,
            edgecolor=HIGHLIGHT_COLOR if i == 4 else EDGE_COLOR,
            facecolor=BOX_COLOR,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Add Text
        ax.text(x_pos, y_positions[i], step, 
                color=TEXT_COLOR, 
                ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Draw Arrow (if not last)
        if i < len(steps) - 1:
            ax.annotate('', 
                xy=(x_pos, y_positions[i+1] + box_height/2), 
                xytext=(x_pos, y_positions[i] - box_height/2),
                arrowprops=dict(arrowstyle='-|>', color=ARROW_COLOR, lw=2, mutation_scale=20)
            )

    plt.title("End-to-End System Workflow: ANN-HWR", 
              fontsize=18, fontweight='bold', color=TEXT_COLOR, pad=20)
    
    plt.savefig('assets/overall_workflow.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print("Workflow image generated successfully in assets/ folder.")

if __name__ == "__main__":
    draw_workflow()
