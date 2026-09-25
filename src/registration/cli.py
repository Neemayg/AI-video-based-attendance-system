"""
Command Line Interface (Entry Point).
"""
import sys
import re
from . import capture
from . import embedding
from . import storage

def main():
    print("========================================")
    print(" AI Video Attendance — Registration")
    print("========================================")
    
    student_id = input("Student ID: ").strip()
    if not student_id:
        print("Error: Student ID cannot be empty.")
        return
        
    name = input("Student Name: ").strip()
    if not name:
        print("Error: Student Name cannot be empty.")
        return
        
    gallery = storage.load_gallery()
    if student_id in gallery:
        print(f"\nStudent ID already exists.")
        print(f"Existing record:\n{student_id}\n{gallery[student_id]['name']}")
        choice = input("\nChoose:\n[A] Cancel\n[B] Re-register / replace\n> ").strip().upper()
        if choice != 'B':
            print("Registration cancelled.")
            return

    # Pre-load models
    print("\nLoading models... (this might take a moment)")
    from .preprocessing import get_mtcnn
    get_mtcnn()
    embedding.get_resnet()
    
    try:
        tensors = capture.capture_face_samples()
        if len(tensors) == 0:
            print("No samples captured. Registration aborted.")
            return
            
        print("\nGenerating embeddings...")
        embeddings = embedding.generate_embeddings(tensors)
        centroid = embedding.create_centroid(embeddings)
        
        storage.save_registration(student_id, name, embeddings, centroid)
        
        print("\n✓ Registration successful")
        print("\nStudent:")
        print(name)
        print("\nID:")
        print(student_id)
        print("\nSamples:")
        print(len(tensors))
        print("\nEmbedding dimension:")
        print(embeddings.shape[1] if len(embeddings.shape) > 1 else 0)
        
    except Exception as e:
        print(f"\nError during registration: {e}")

if __name__ == "__main__":
    main()
