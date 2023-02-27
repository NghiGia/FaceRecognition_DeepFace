    # img_region = [0, 0, image.shape[1], image.shape[0]]
    # img_objs = [(image, img_region, 0)]
    # for img, region, _ in img_objs:
    #     # custom normalization
    #     img = functions.normalize_input(img=img, normalization=normalization)

    #     # represent
    #     if "keras" in str(type(model)):
    #         # new tf versions show progress bar and it is annoying
    #         embedding = model.predict(img, verbose=0)[0].tolist()
    #     else:
    #         # SFace and Dlib are not keras models and no verbose arguments
    #         embedding = model.predict(img)[0].tolist()

    #     resp_obj = {}
    #     resp_obj["embedding"] = embedding
    #     resp_obj["facial_area"] = region
    #     resp_objs.append(resp_obj)