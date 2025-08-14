import os

def save_graph_image(graph, 
                     output_dir: str = "output", 
                     filename: str = "graph.png"):
    """워크플로우 그래프 이미지를 저장합니다."""
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, filename)
    
    with open(image_path, "wb") as f:
        f.write(graph_image)
    
    print(f"그래프 이미지가 저장되었습니다: {image_path}")
    return image_path