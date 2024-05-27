import os, json, argparse
from pdf2image import convert_from_path

import numpy as np
from tqdm import tqdm

def convert_due(droot, dset):
	dset_dict = {
		"data_dir": os.path.join(droot, dset)
	}
	os.makedirs(os.path.join(droot, dset, "images"), exist_ok=True)

	print(f"READ {dset} DATA from {dset_dict['data_dir']} and convert PDF to PNG 300 dpi...")
	for _split in ["val", "test", "train"]:
		print(f"split: {_split}")
		documents_content = {}
		saved, broken = 0, 0
		for jl in tqdm(list(open(os.path.join(dset_dict["data_dir"], _split, "documents_content.jsonl")))):
			jline = json.loads(jl)
			try:
				images = convert_from_path(os.path.join(dset_dict["data_dir"], "pdfs", jline['name'] + '.pdf'), dpi=300)
				images[0].save(os.path.join(dset_dict["data_dir"], "images", f"{jline['name']}.png"), 'PNG')
				saved += 1
			except Exception as e:
				print(f"{e}, broken file: {os.path.join(dset_dict['data_dir'], 'pdfs', jline['name'])}")
				broken += 1
				continue
			w, h = images[0].size
			ocr_normalized_boxes = []
			for _bbox in jline['contents'][0]['tokens_layer']['positions']:
				_bbn = [
					_bbox[0]/w, _bbox[1]/h, _bbox[2]/w, _bbox[3]/h  # CORRECT ORDER: left, top, right, bottom
				]
				if np.any(np.array(_bbn) > 1):
					_bbn = np.clip(_bbn, 0, 1).tolist()
				ocr_normalized_boxes.append(_bbn)
			documents_content[jline['name']] = {
				"text": jline['contents'][0]['text'],
				"ocr_tokens": jline['contents'][0]['tokens_layer']['tokens'],
				"ocr_boxes": jline['contents'][0]['tokens_layer']['positions'],
				"ocr_normalized_boxes": ocr_normalized_boxes,
				"width": w,
				"height": h,
			}
		assert len(documents_content) == saved
		print(f"{_split} no. broken/saved : {broken}/{saved}")
		dset_dict[_split] = {
			"documents_content": documents_content, # list of document content (image name, orc tokens, ...)
			"document": [json.loads(jl)
				for jl in list(open(os.path.join(dset_dict["data_dir"], _split, "document.jsonl")))
			] # list of documents, each contains a lisg of (q,a)s
		}
		if _split != "train":
			with open(os.path.join(dset_dict["data_dir"], _split, 'ssl.json'), 'w') as f:
				list_docs = []
				for _k, _v in dset_dict[_split]["documents_content"].items():
					_v.update({"document_id": _k, "dataset": dset})
					list_docs.append(_v)
				json.dump(list_docs, f)

	print("SAVE npy file for Doc VQA task...")
	for _split in ["val", "test", "train"]:
		print(f"split: {_split}")
		records = []
		for _doc in tqdm(dset_dict[_split]["document"]):
			for _anno in _doc['annotations']:
				answers = [_anno['values'][0]['value']]
				if 'value_variants' in _anno['values'][0]:
					answers += _anno['values'][0]['value_variants']
				metadata = {}
				if 'metadata' in _anno:
					metadata = _anno['metadata']
				if 'metadata' in _doc:
					metadata = _doc['metadata']
				metadata.update({"dataset": dset, "id": _anno['id']})
				if _doc['name'] in dset_dict[_split]["documents_content"]:
					doc_content = dset_dict[_split]["documents_content"][_doc['name']]
					records.append({
						"metadata": metadata,
						"question": _anno['key'],
						"answers": answers,
						"image_name": _doc['name'],
						"image_height": doc_content["height"],
						"image_width": doc_content["width"],
						"ocr_tokens": doc_content['ocr_tokens'],
						"ocr_normalized_boxes": doc_content['ocr_normalized_boxes'],
					})
		np.save(open(os.path.join(dset_dict["data_dir"], _split, 'vqa.npy'), 'wb'), np.array(records), allow_pickle=True)

		if _split == "train":
			doc2inds = {}
			for _ind, _record in enumerate(records):
				if _record["image_name"] not in doc2inds:
					doc2inds[_record["image_name"]] = [_ind]
				else:
					doc2inds[_record["image_name"]].append(_ind)
			assert sum([len(_v) for _v in doc2inds.values()]) == len(records)

			with open(os.path.join(dset_dict["data_dir"], _split, 'ssl.json'), 'w') as f:
				list_docs = []
				for _k, _v in dset_dict[_split]["documents_content"].items():
					_v.update({"document_id": _k, "dataset": dset, "vqa_indices": doc2inds[_k]})
					list_docs.append(_v)
				json.dump(list_docs, f)
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/data/users/vkhanh/due', help='path/to/due/benchmark')
	parser.add_argument('--dataset', type=str, default="docvqa", help='dataset')

	args = parser.parse_args()
	dset = args.dataset
	data_dir = args.data_dir
	assert dset.lower() in ['deepform', 'docvqa', 'infovqa', 'tabfact', 'wtq']
	convert_due(data_dir, dset)