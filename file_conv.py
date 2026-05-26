# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:45:25 2026

@author: fowak

File converter to PDF
"""

import os
import subprocess
import fitz
    
    
def unified_document_pipeline(folder_path: str, final_output_name: str):
    
    # --- STAGE 1: The LibreOffice Headless Converter (Keep this exactly the same) ---
    print(f"\n[STAGE 1] Batch converting Word documents to PDF in: {folder_path}")
    libreoffice_exe = r"C:\Program Files\LibreOffice\program\soffice.exe"
    
    if not os.path.exists(libreoffice_exe):
        print("\n[CRITICAL ERROR] LibreOffice not found. Please install it to proceed.")
        return

    all_files = os.listdir(folder_path)
    word_docs = [file for file in all_files if file.endswith((".doc", ".docx"))]
    
    for doc in word_docs:
        print(f"  -> Converting: {doc}")
        full_doc_path = os.path.join(folder_path, doc)
        command = [libreoffice_exe, "--headless", "--convert-to", "pdf", "--outdir", folder_path, full_doc_path]
        subprocess.run(command, check=True)
        
    print("[STAGE 1 COMPLETE] Individual PDFs generated.")

    # --- STAGE 2: The PyMuPDF Bulldozer ---
    print("\n[STAGE 2] Sweeping folder for PDFs to merge using PyMuPDF...")
    
    # Create an empty master PDF object
    master_pdf = fitz.open()

    all_files = os.listdir(folder_path)
    pdf_files = [file for file in all_files if file.endswith(".pdf")]
    pdf_files.sort()

    print(f"Found {len(pdf_files)} PDFs. Stitching together...")

    for pdf in pdf_files:
        full_path = os.path.join(folder_path, pdf)
        try:
            # 1. Open the individual PDF
            with fitz.open(full_path) as doc:
                # 2. Forcibly insert all pages into the master PDF
                master_pdf.insert_pdf(doc)
            print(f"  -> Appended: {pdf}")
        except Exception as e:
            # If PyMuPDF can't open it, the file is truly, completely dead.
            print(f"  [CRITICAL] FAILED on {pdf}. Reason: {e}")
            continue

    # 3. Write the final master file
    master_pdf.save(final_output_name)
    master_pdf.close()
    
    print(f"\n[PIPELINE SUCCESS] Master document compiled as: {final_output_name}")

# --- EXECUTION ---
target_folder = r"C:\Users\fowak\Documents\Work\AI Engineering\RAG Pipeline\RAG app\Injection Data\KIA Manuals"
output_file = "2023 Kia Forte Manual Data.pdf"  

unified_document_pipeline(target_folder, output_file)
    