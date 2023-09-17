

class_obj = {
    "class": "Image",
    "description": "Base image",  # description of the class
    "properties": [
        {
            "dataType": ["text"],
            "name": "desc",
            "description": "The auto-generated title",
        },
        {
            "dataType": ["text"],
            "description": "The external ref",
            "name": "external_ref",
        },
        {
            "dataType": ["ImageCLIP"],
            "description": "The external ref",
            "name": "vec_clip_ref",
        }
    ],    
    "vectorizer": "none",  
    "moduleConfig": {
    }
}


class_obj_img_clip = {
    "class": "ImageCLIP",
    "description": "CLIP description of the image", 
    "properties": [
        {
            "dataType": ["Image"],
            "description": "The base image class",
            "name": "image_ref",
        },
        {
            "dataType": ["text"],
            "description": "model version metadata",
            "name": "modelVersion",
        }
    ],    
    "vectorizer": "none", # We use custom ingenstion 
    "moduleConfig": {
    }
}