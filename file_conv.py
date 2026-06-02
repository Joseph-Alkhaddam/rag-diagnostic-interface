# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:45:25 2026
@author: fowak

Batch File Conversion & PDF Merging Utility
"""

import os
import subprocess
import fitz  # PyMuPDF


def convert_word_to_pdf(
    folder_path: str, 
    libreoffice_exe: str = r"C:\Program Files\LibreOffice\program\soffice.exe"
) -> None:
    """
    Scans a directory for Word documents and batch converts them to PDF.

    Executes a headless instance of LibreOffice via the system shell to process 
    .doc and .docx files, outputting the resulting PDFs into the same directory.

    Parameters
    ----------
    folder_path : str
        The absolute or relative path to the directory containing the Word documents.
    libreoffice_exe : str, optional
        The absolute path to the local LibreOffice executable file.

    Returns
    -------
    None
    """
    print(f"\n[STAGE 1] Batch converting Word documents to PDF in: {folder_path}")
    
    if not os.path.exists(libreoffice_exe):
        print("\n[CRITICAL ERROR] LibreOffice not found. Please verify the executable path.")
        return

    try:
        all_files = os.listdir(folder_path)
    except Exception as e:
        print(f"[ERROR] Could not read target directory: {folder_path}. Details: {e}")
        return

    word_docs = [file for file in all_files if file.endswith((".doc", ".docx"))]
    
    if not word_docs:
        print("  -> No Word documents found. Skipping conversion.")
        return

    for doc in word_docs:
        print(f"  -> Converting: {doc}")
        full_doc_path = os.path.join(folder_path, doc)
        command = [
            libreoffice_exe, 
            "--headless", 
            "--convert-to", 
            "pdf", 
            "--outdir", 
            folder_path, 
            full_doc_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to convert {doc}. Details: {e.stderr.decode()}")
            continue
            
    print("[STAGE 1 COMPLETE] Document conversion cycle finished.")


def merge_directory_pdfs(folder_path: str, final_output_name: str) -> None:
    """
    Compiles all PDF files within a specified directory into a single master document.

    Utilizes PyMuPDF to sequentially append the binary page data of each PDF 
    found in the target folder into a unified output file.

    Parameters
    ----------
    folder_path : str
        The absolute or relative path to the directory containing the PDF files.
    final_output_name : str
        The designated filename or path for the compiled master PDF.

    Returns
    -------
    None
    """
    print("\n[STAGE 2] Sweeping folder for PDFs to merge using PyMuPDF...")
    
    try:
        all_files = os.listdir(folder_path)
    except Exception as e:
        print(f"[ERROR] Could not read target directory: {folder_path}. Details: {e}")
        return

    pdf_files = [file for file in all_files if file.endswith(".pdf")]
    pdf_files.sort()

    if not pdf_files:
        print("  -> No PDF files found to merge.")
        return

    print(f"Found {len(pdf_files)} PDFs. Stitching together...")
    
    master_pdf = fitz.open()

    for pdf in pdf_files:
        full_path = os.path.join(folder_path, pdf)
        try:
            with fitz.open(full_path) as doc:
                master_pdf.insert_pdf(doc)
            print(f"  -> Appended: {pdf}")
        except Exception as e:
            print(f"  [CRITICAL] FAILED on {pdf}. Reason: {str(e)}")
            continue

    try:
        master_pdf.save(final_output_name)
        print(f"\n[PIPELINE SUCCESS] Master document compiled as: {final_output_name}")
    except Exception as e:
        print(f"\n[ERROR] Could not save master PDF. Details: {str(e)}")
    finally:
        master_pdf.close()


def unified_document_pipeline(folder_path: str, final_output_name: str) -> None:
    """
    Orchestrates the complete document ingestion pipeline.
    First converts any localized Word documents, then merges all resulting PDFs.
    """
    convert_word_to_pdf(folder_path)
    merge_directory_pdfs(folder_path, final_output_name)


# --- ENTRY POINT FOR STANDALONE EXECUTION ---
if __name__ == "__main__":
    # Define your execution parameters here so they don't clutter the logic above
    TARGET_FOLDER = r"C:\Users\fowak\Documents\Work\AI Engineering\RAG Pipeline\Injection Data\Building Manuals"
    OUTPUT_FILE = "Ontario Building Code 2024.pdf"  
    
    # Execute the master pipeline
    unified_document_pipeline(TARGET_FOLDER, OUTPUT_FILE)