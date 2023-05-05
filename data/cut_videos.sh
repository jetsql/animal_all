IN_DATA_DIR="./ava/videos"

OUT_DATA_DIR="./ava/videos_cut"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi
 
for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 0 -t 4 -i "${video}" -r 30 -strict experimental "${out_name}"
  fi
done
