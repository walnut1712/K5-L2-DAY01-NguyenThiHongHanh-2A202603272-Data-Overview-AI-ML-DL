# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:*06 November 2026*

**Runtime Colab:** CPU/GPU

Python 3 Google Compute Engine backend (GPU)

Showing resources since 7:09 PM

System RAM
8.5 / 12.7 GB
 
GPU RAM
0.2 / 15.0 GB
 
Disk
0.2 / 112.6 GB

**Python / PyTorch / Ultralytics:* 3.13.16 / 2.11.0+cu130 / 8.4.145*

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- This record describes the whole image. It has a single class_name (cab) and a single score, and it has no bbox or polygon coordinates saying where an object is. One label is assigned to the entire traffic image, even though the scene contains other things besides taxis.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- The class list comes with the taxonomy and checkpoint, not from the model inventing it. yolo11n-cls.pt was pretrained on ImageNet-1K, so it can only choose from that taxonomy’s 1000 fixed classes.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- class_id is a stable numeric code for machines to process and compare.
- class_name lets humans read the label quickly.
- taxonomy_name says which class table the ID belongs to. The same number (for example 468) can mean a different class in another taxonomy such as COCO-80, and the same name can be defined differently across taxonomies.
- If any one of the three is missing, the label can be ambiguous or impossible to reproduce.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- The guideline must give a rule for choosing exactly one class, for example:
  - pick the main subject, the largest object, or the most prominent one;
  - a tie-break rule when two subjects are about equal;
  - when to use an “unclear” label or escalate to the Lab Coach.
- Without this, two annotators could pick different classes for the same image.
6. Vì sao không được lấy `score` của model làm ground truth?
- The score is only the model’s confidence in its own prediction. It is not a correct label made by a person following a guideline. A score of 0.51 means the model is only moderately sure, and it can still be wrong. Here the image is a COCO scene classified with an ImageNet taxonomy, so the top-1 label may not match what a human annotator would choose. Ground truth must be created by people and checked by a reviewer.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- The box starts at the top-left corner at x = 385.33, y = 69.24 pixels, measured from the image’s top-left origin (0, 0) in a 640 × 427 image. It ends at the bottom-right corner at x = 498.92, y = 348.92. So it is about 113.58 pixels wide and 279.68 pixels tall. It is much taller than wide, which fits a standing person seen from behind, and it covers the man from head to feet.
- The width and height check out: 498.92 − 385.33 ≈ 113.58 and 348.92 − 69.24 = 279.68. The small difference in the width is just rounding.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- At 0.20, there are 17 predictions. They include weak extras such as spoon (×3), potted plant, dining table and bottle.
- At 0.60, there are 6 predictions: person, bowl, bowl, oven, oven, person.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- A lower threshold keeps more boxes (here 6 → 17), so the reviewer has more to check. For each extra box they must decide whether it is a real object or a false positive, delete the wrong ones, fix wrong class names, and tighten boxes that don’t fit. They must also look for duplicates or overlapping boxes on the same object. Lowering the threshold doesn’t make labels more correct. It only adds review work, and the model score still isn’t proof that a box is right.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Draw one tight rectangle around the full visible extent of each object instance, touching its outermost visible edges without including background, and use a separate box for each object even when objects overlap.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- On the left edge of kitchen, only a hand and forearm are visible, and the rest of the person is cut off by the image border. The box starts at x = 0.12, which is right at the image edge, and its size is only 61.44 × 47.59 pixels. It also overlaps a bowl box and the oven box, which makes it harder to read. I would escalate to the Lab Coach instead of writing a rule myself. The guideline doesn’t say whether a lone hand or arm counts as a “person”, or how much of an object must be visible before it gets a box. Annotators could decide this differently, so guessing would make the labels inconsistent. Once the Lab Coach decides, it can be added to the guideline as a rule.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- The box [385.45, 66.44, 498.02, 348.58] is only a rectangle around the person, so it also contains background such as the hanging pans, the towel and the wall. The polygon uses 348 points to trace the actual outline: the curly hair, the arms bent at the hips, the apron and the legs. It tells us which pixels belong to the person and which don’t, and so gives the real shape, the visible area and the boundary, none of which a box can show.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id (kitchen-001) is only a label for one specific object in this lab’s output, so that each polygon can be told apart and referred to, even when two objects have the same class. For example, the two ovens or the several bowls each get their own ID and polygon. It is not a class ID, because class_id (here 0) says what kind of object it is (person) and is shared by every person in the taxonomy, while instance_id identifies one individual. It is also not a tracking ID, because it doesn’t follow the same object across video frames. It is just a number given within a single image, and it can change if the model or the order of predictions changes.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Draw the polygon along the outer visible edge of the object, so that it includes the whole object but excludes background, and place points densely enough on curves (hair, hands, folds of clothing) that the mask follows the true outline. If an object is partly hidden by another, trace only the visible part and do not guess the hidden part.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- In kitchen, the dining table 0.58 mask covers the whole tabletop, including the jar, the plates, the dough and the bowls sitting on it, while the bowls have their own masks (bowl 0.70, bowl 0.74). So the same pixels are inside two masks, and the table’s mask edge is hard to place where objects touch it. 
- I would use a guideline rule: each object gets its own mask for its visible pixels, and an object lying on top of the table is cut out of the table’s mask, so no pixel belongs to two instances. This is a repeatable case that a rule can settle. I would escalate only what the rule can’t decide, such as whether a surface covered with dough should count as “dining table” at all.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | One class label for the whole image: class_id, class_name and taxonomy_name (such as: 468, cab, ImageNet-1K). No location information. | In traffic, the model’s top-1 is cab with score 0.51, which is only moderately confident. The image is a COCO street scene with several subjects but ImageNet gives one label. The guideline doesn’t say which subject to pick. | Picks exactly one class from the taxonomy by the guideline’s rule (such as: the main subject), and escalates when two subjects are about equal. | That the label is in the taxonomy, matches the guideline’s rule for choosing the main subject, and was not copied from the model score. |
| Phát hiện vật thể | One tight rectangle per object, in pixel xyxy, with a class (such as: person [385.33, 69.24, 498.92, 348.92], 113.58 × 279.68 px). | In kitchen, person 0.61 at [0.12, 263.18, 61.55, 310.78] is only a hand and forearm cut off by the left image edge, and it overlaps bowl and oven boxes. The threshold also changes the count a lot (6 at 0.60, 11 at 0.35, 17 at 0.20), with weak extras like spoon and bottle at 0.20. | Draws one tight box around each visible object, one box per instance even when they overlap. For partly visible or ambiguous objects, follows the guideline or escalates to the Lab Coach. | That boxes are tight and have the right class, that no objects are missed, that false positives and duplicates are removed, and that cut-off objects are handled consistently. |
| Instance segmentation | One polygon per object instance (a list of [x, y] points), with a class and its own instance_id (such as: kitchen-001, person, 348 points). | In kitchen, the dining table 0.58 mask covers the whole tabletop including the jar, plates, dough and bowls, so those pixels also lie inside the bowl masks (bowl 0.70, bowl 0.74). The potted plant 0.63 mask covers hanging dried leaves, which fits no clear COCO-80 class. | Traces the visible outline of each object with enough points on curves, cuts overlapping objects out of one another’s masks, and escalates objects that don’t fit any class. | That masks follow the true edges without leaking into the background or neighbours or cutting off parts, that overlapping instances have separate masks and IDs, and that the class is correct. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- I only use the three public COCO sample images provided by the notebook (traffic, kitchen, dining), and I never upload personal photos or internal data, or put any personal information (name, student ID, email, phone number) in the report or the output files.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- I would stop working with that data right away, not process, upload or share it, and report it to the Lab Coach (my instructor or mentor) so they can decide what to do. I would not make up labels or keep the data in my repo.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
