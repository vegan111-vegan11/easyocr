import cv2
def detect(self, img, min_size=20, text_threshold=0.7, low_text=0.4, \
           link_threshold=0.4, canvas_size=2560, mag_ratio=1., \
           slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
           width_ths=0.5, add_margin=0.1, reformat=True, optimal_num_chars=None,
           threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0,
           ):
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr detext 함수 들어옴 self: {self}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr detext 함수 들어옴 reformat: {reformat}')

    if reformat:
        img, img_cv_grey = reformat_input(img)

    text_box_list = self.get_textbox(self.detector,
                                     img,
                                     canvas_size=canvas_size,
                                     mag_ratio=mag_ratio,
                                     text_threshold=text_threshold,
                                     link_threshold=link_threshold,
                                     low_text=low_text,
                                     poly=False,
                                     device=self.device,
                                     optimal_num_chars=optimal_num_chars,
                                     threshold=threshold,
                                     bbox_min_score=bbox_min_score,
                                     bbox_min_size=bbox_min_size,
                                     max_candidates=max_candidates,
                                     )

    horizontal_list_agg, free_list_agg = [], []
    for text_box in text_box_list:
        horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                    ycenter_ths, height_ths,
                                                    width_ths, add_margin,
                                                    (optimal_num_chars is None))
        if min_size:
            horizontal_list = [i for i in horizontal_list if max(
                i[1] - i[0], i[3] - i[2]) > min_size]
            free_list = [i for i in free_list if max(
                diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
        horizontal_list_agg.append(horizontal_list)
        free_list_agg.append(free_list)

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr detext 함수 들어옴 horizontal_list_agg: {horizontal_list_agg}')

    return horizontal_list_agg, free_list_agg
def readtext(self, image, decoder='greedy', beamWidth=5, batch_size=1, \
             workers=0, allowlist=None, blocklist=None, detail=1, \
             rotation_info=None, paragraph=False, min_size=20, \
             contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, \
             text_threshold=0.7, low_text=0.4, link_threshold=0.4, \
             canvas_size=2560, mag_ratio=1., \
             slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
             width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1,
             threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0,
             output_format='standard'):
    '''
    Parameters:
    image: file path or numpy-array or a byte stream object
    '''

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr readtext 함수 image 여기서 에러???? image :  ')

    img, img_cv_grey = reformat_input(image)

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr readtext 함수 img : {img}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr readtext 함수 img_cv_grey : {img_cv_grey}')

    # self.
    horizontal_list, free_list = self.detect(img,
                                             min_size=min_size, text_threshold=text_threshold, \
                                             low_text=low_text, link_threshold=link_threshold, \
                                             canvas_size=canvas_size, mag_ratio=mag_ratio, \
                                             slope_ths=slope_ths, ycenter_ths=ycenter_ths, \
                                             height_ths=height_ths, width_ths=width_ths, \
                                             add_margin=add_margin, reformat=False, \
                                             threshold=threshold, bbox_min_score=bbox_min_score, \
                                             bbox_min_size=bbox_min_size, max_candidates=max_candidates
                                             )
    # get the 1st result from hor & free list as self.detect returns a list of depth 3
    horizontal_list, free_list = horizontal_list[0], free_list[0]

    print(
        f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr result 여기서 에러 readtext 함수 호출전 여기서 에러남????  함수에서 에러???>????? horizontal_list : {horizontal_list}')

    result = self.recognize(img_cv_grey, horizontal_list, free_list, \
                            decoder, beamWidth, batch_size, \
                            workers, allowlist, blocklist, detail, rotation_info, \
                            paragraph, contrast_ths, adjust_contrast, \
                            filter_ths, y_ths, x_ths, False, output_format)
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr result 여기서 에러 readtext 함수에서 에러???>?????  : {result}')

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!easyocr readtext 함수 horizontal_list : {horizontal_list}')
    return result

if __name__ == '__main__':
    print('main 함수 들어옴')
