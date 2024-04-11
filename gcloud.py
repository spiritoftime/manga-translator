from google.cloud import vision
from PIL import Image
from paddleocr import draw_ocr


img_path = 'jap-raw.jpeg'
response = None
def detect_text(path):
    global response
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
def draw_result(img_path):
    image = Image.open(img_path).convert('RGB')
    boxes = []
    for text in response.text_annotations:
        vertices = []
        for vertex in text.bounding_poly.vertices:
            vertices.append([vertex.x, vertex.y])
        boxes.append(vertices)
    print(boxes,'boxes')
    txts = [line.description for line in response.text_annotations]
    print(txts,'txts')
    print(len(boxes),len(txts))
    im_show = draw_ocr(image, boxes, txts,  font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result_g_cloud.jpg')

detect_text(img_path)
draw_result(img_path)
