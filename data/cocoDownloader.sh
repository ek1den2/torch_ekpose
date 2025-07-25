mkdir coco2017
mkdir coco2017/images
cd coco2017

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip test2017.zip -d images/

mv images/train2017 images/train
mv images/val2017 images/val
mv images/test2017 images/test

unzip annotations_trainval2017.zip
unzip image_info_test2017.zip

mv annotations/person_keypoints_train2017.json annotations_train.json
mv annotations/person_keypoints_val2017.json annotations_val.json
mv annotations/image_info_test-dev2017.json annotations_test.json

rm -rf annotations

rm -f train2017.zip
rm -f val2017.zip
rm -f test2017.zip
rm -f annotations_trainval2017.zip
rm -f image_info_test2017.zip