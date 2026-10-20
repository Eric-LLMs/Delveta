# Split Integrity Report (preliminary)

The independent validator regenerates this report from the output files.

| Group | Target train/test/val | Actual train/test/val |
|---|---|---|
| REJECT|en | {'train': 1125, 'test': 300, 'val': 75} | {'train': 1125, 'test': 300, 'val': 75} |
| REJECT|zh | {'train': 1125, 'test': 300, 'val': 75} | {'train': 1125, 'test': 300, 'val': 75} |
| cap-add-term|en | {'train': 1456, 'test': 388, 'val': 97} | {'train': 1456, 'test': 388, 'val': 97} |
| cap-add-term|mixed | {'train': 63, 'test': 17, 'val': 4} | {'train': 63, 'test': 17, 'val': 4} |
| cap-add-term|zh | {'train': 1398, 'test': 373, 'val': 93} | {'train': 1399, 'test': 372, 'val': 93} |
| cap-artifact|en | {'train': 698, 'test': 186, 'val': 47} | {'train': 696, 'test': 189, 'val': 46} |
| cap-artifact|mixed | {'train': 276, 'test': 73, 'val': 18} | {'train': 276, 'test': 73, 'val': 18} |
| cap-artifact|zh | {'train': 424, 'test': 113, 'val': 28} | {'train': 424, 'test': 113, 'val': 28} |
| cap-bash|en | {'train': 697, 'test': 186, 'val': 47} | {'train': 697, 'test': 186, 'val': 47} |
| cap-bash|mixed | {'train': 227, 'test': 60, 'val': 15} | {'train': 227, 'test': 60, 'val': 15} |
| cap-bash|zh | {'train': 472, 'test': 126, 'val': 32} | {'train': 472, 'test': 126, 'val': 32} |
| cap-create-folder|en | {'train': 1491, 'test': 398, 'val': 99} | {'train': 1466, 'test': 402, 'val': 120} |
| cap-create-folder|mixed | {'train': 224, 'test': 60, 'val': 15} | {'train': 225, 'test': 59, 'val': 15} |
| cap-create-folder|zh | {'train': 1337, 'test': 356, 'val': 89} | {'train': 1337, 'test': 356, 'val': 89} |
| cap-edit-file|en | {'train': 698, 'test': 186, 'val': 47} | {'train': 697, 'test': 187, 'val': 47} |
| cap-edit-file|mixed | {'train': 192, 'test': 51, 'val': 13} | {'train': 192, 'test': 51, 'val': 13} |
| cap-edit-file|zh | {'train': 507, 'test': 135, 'val': 34} | {'train': 508, 'test': 134, 'val': 34} |
| cap-mindmap|en | {'train': 697, 'test': 186, 'val': 46} | {'train': 695, 'test': 189, 'val': 45} |
| cap-mindmap|mixed | {'train': 22, 'test': 6, 'val': 1} | {'train': 22, 'test': 6, 'val': 1} |
| cap-mindmap|zh | {'train': 675, 'test': 180, 'val': 45} | {'train': 674, 'test': 182, 'val': 44} |
| cap-pdf-extract-text|en | {'train': 546, 'test': 146, 'val': 36} | {'train': 546, 'test': 146, 'val': 36} |
| cap-pdf-extract-text|mixed | {'train': 336, 'test': 90, 'val': 22} | {'train': 335, 'test': 92, 'val': 21} |
| cap-pdf-extract-text|zh | {'train': 511, 'test': 136, 'val': 34} | {'train': 511, 'test': 136, 'val': 34} |
| cap-pdf-table-to-text|en | {'train': 547, 'test': 146, 'val': 36} | {'train': 547, 'test': 146, 'val': 36} |
| cap-pdf-table-to-text|mixed | {'train': 285, 'test': 76, 'val': 19} | {'train': 282, 'test': 81, 'val': 17} |
| cap-pdf-table-to-text|zh | {'train': 562, 'test': 150, 'val': 37} | {'train': 562, 'test': 150, 'val': 37} |
| cap-rag-search|en | {'train': 1307, 'test': 348, 'val': 87} | {'train': 1307, 'test': 348, 'val': 87} |
| cap-rag-search|mixed | {'train': 399, 'test': 106, 'val': 27} | {'train': 396, 'test': 111, 'val': 25} |
| cap-rag-search|zh | {'train': 1212, 'test': 323, 'val': 81} | {'train': 1212, 'test': 323, 'val': 81} |
| cap-read-document|en | {'train': 548, 'test': 146, 'val': 37} | {'train': 547, 'test': 147, 'val': 37} |
| cap-read-document|mixed | {'train': 192, 'test': 51, 'val': 13} | {'train': 190, 'test': 55, 'val': 11} |
| cap-read-document|zh | {'train': 658, 'test': 175, 'val': 44} | {'train': 658, 'test': 175, 'val': 44} |
| cap-read-file|en | {'train': 698, 'test': 186, 'val': 47} | {'train': 695, 'test': 191, 'val': 45} |
| cap-read-file|mixed | {'train': 245, 'test': 65, 'val': 16} | {'train': 245, 'test': 65, 'val': 16} |
| cap-read-file|zh | {'train': 456, 'test': 121, 'val': 30} | {'train': 456, 'test': 121, 'val': 30} |
| cap-research|en | {'train': 300, 'test': 80, 'val': 20} | {'train': 301, 'test': 79, 'val': 20} |
| cap-research|mixed | {'train': 9, 'test': 3, 'val': 1} | {'train': 9, 'test': 3, 'val': 1} |
| cap-research|zh | {'train': 291, 'test': 77, 'val': 19} | {'train': 291, 'test': 77, 'val': 19} |
| cap-slides|en | {'train': 547, 'test': 146, 'val': 37} | {'train': 546, 'test': 146, 'val': 38} |
| cap-slides|mixed | {'train': 302, 'test': 80, 'val': 20} | {'train': 301, 'test': 82, 'val': 19} |
| cap-slides|zh | {'train': 546, 'test': 146, 'val': 36} | {'train': 546, 'test': 146, 'val': 36} |
| cap-social-search|en | {'train': 1456, 'test': 388, 'val': 97} | {'train': 1452, 'test': 396, 'val': 93} |
| cap-social-search|mixed | {'train': 697, 'test': 186, 'val': 47} | {'train': 696, 'test': 188, 'val': 46} |
| cap-social-search|zh | {'train': 1140, 'test': 304, 'val': 76} | {'train': 1142, 'test': 303, 'val': 75} |
| cap-summary|en | {'train': 547, 'test': 146, 'val': 36} | {'train': 547, 'test': 146, 'val': 36} |
| cap-summary|mixed | {'train': 151, 'test': 40, 'val': 10} | {'train': 149, 'test': 43, 'val': 9} |
| cap-summary|zh | {'train': 696, 'test': 186, 'val': 46} | {'train': 696, 'test': 186, 'val': 46} |
| cap-translate|en | {'train': 1324, 'test': 353, 'val': 88} | {'train': 1324, 'test': 353, 'val': 88} |
| cap-translate|mixed | {'train': 157, 'test': 42, 'val': 10} | {'train': 152, 'test': 50, 'val': 7} |
| cap-translate|zh | {'train': 1468, 'test': 392, 'val': 98} | {'train': 1468, 'test': 392, 'val': 98} |
| cap-vision|en | {'train': 550, 'test': 147, 'val': 37} | {'train': 550, 'test': 147, 'val': 37} |
| cap-vision|mixed | {'train': 164, 'test': 44, 'val': 11} | {'train': 164, 'test': 48, 'val': 7} |
| cap-vision|zh | {'train': 686, 'test': 183, 'val': 46} | {'train': 686, 'test': 183, 'val': 46} |
| cap-web-search|en | {'train': 1307, 'test': 349, 'val': 87} | {'train': 1305, 'test': 352, 'val': 86} |
| cap-web-search|mixed | {'train': 270, 'test': 72, 'val': 18} | {'train': 269, 'test': 75, 'val': 16} |
| cap-web-search|zh | {'train': 1338, 'test': 357, 'val': 89} | {'train': 1338, 'test': 357, 'val': 89} |
