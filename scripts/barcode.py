import cv2
from pyzbar.pyzbar import decode

class BarcodeDecoder:
    def decode(self,image):
        barcodes = decode(image)
        
        if barcodes:
            for barcode in barcodes:
                # Extract the barcode data and format
                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                
                return barcode_data
        else:
            return ("No barcode detected.")