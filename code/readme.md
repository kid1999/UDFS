
# Project Data Description

## Data Structure

  - All raw data is stored in a single folder, for example, `data/raw/`.
  - The data file naming convention is: `label_xxx.pcap`
      - `label` represents the class label.
      - For example, `1_qq.com_20250519133127.pcap` represents the complete traffic for a single visit (trace) to qq.com.

Example:

```
data/raw/
├── 0_001.pcap
├── 0_002.pcap
├── 1_001.pcap
├── 2_001.pcap
...
```

## Dataset Construction

This project provides a `build_dataset.py` script to convert the raw data into the dataset format required for model training.

### Usage

```python
if __name__ == '__main__':
    # ---------- Example ----------
    folder = r"F:\dataset\baidu20\20250823"
    print(folder)
    df = build_dataset_from_folder_multithread(folder, payload_only=True, max_workers=8)
    df.to_pickle("baidu20_0823.pkl")
```

## Notes

1.  Please ensure that the raw data filenames follow the `label_xxx` format.

## Dataset Descriptions

> Six pre-processed datasets are available in the `datasets` directory.

1.  **`baidu20_0817.pkl`**: Contains the complete traffic traces for 100 visits to each of 20 popular websites (e.g., Baidu, Weibo), collected on 2025/08/17.
2.  **`baidu20_0823.pkl`**: Contains the complete traffic traces for 100 visits to each of the same 20 popular websites, collected on 2025/08/23. (Can be used with the 0817 dataset for concept drift validation).
3.  **`github10_20250303.pkl`**: Contains the complete traffic traces for 100 visits to each of 10 popular GitHub repositories (e.g., `github.com/freeCodeCamp/freeCodeCamp`, `github.com/vinta/awesome-python`), collected on 2025/03/03.
4.  **`github10_20250502.pkl`**: Contains the complete traffic traces for approximately 100 visits to each of the same 10 popular repositories, collected on 2025/05/02. (Can be used with the 0303 dataset for concept drift validation).
5.  **`github100.pkl`**: Contains the complete traffic traces for 100 visits to each of 100 popular GitHub repositories (e.g., `github.com/freeCodeCamp/freeCodeCamp`, `github.com/vinta/awesome-python`). (Can be used for open-world recognition experiments).
6.  **`baidu10000.pkl`**: Contains the complete traffic trace for a single visit to each of approximately 10,000 popular websites (e.g., Baidu, Weibo). (Can be used for large-scale or concept drift experiments).




# 项目数据说明

## 数据存放结构

- 所有原始数据存放在一个文件夹下，例如 `data/raw/`。
- 数据文件命名规则为：`label_xxx.pcap`  
  - `label` 表示类别标签  
  - 如：`1_qq.com_20250519133127.pcap` 表示对qq.com的一次访问(trace)的完整流量

示例：
```
data/raw/
├── 0_001.pcap
├── 0_002.pcap
├── 1_001.pcap
├── 2_001.pcap
...
````

## 数据集构建

本项目提供 `build_dataset.py` 脚本，用于将原始数据转换为模型训练所需的数据集格式。

### 使用方法

```python
if __name__ == '__main__':
    # ---------- 示例 ----------
    folder = r"F:\dataset\baidu20\20250823"
    print(folder)
    df = build_dataset_from_folder_multithread(folder, payload_only=True, max_workers=8)
    df.to_pickle("baidu20_0823.pkl")
````

## 注意事项
1. 请确保原始数据命名遵循 `label_xxx` 格式。


## 数据集说明
> 已清洗的六个常用数据集存放在`datasets`下。
1. `baidu20_0817.pkl`包含2025/08/17采集的baidu、weibo等20个热门网站，每个类别访问100次的完整访问流量。
2. `baidu20_0823.pkl`包含2025/08/23采集的baidu、weibo等20个热门网站，每个类别访问100次的完整访问流量。（可配合0817做概念漂移验证）
3. `github10_20250303.pkl`包含20250303采集的github.com/freeCodeCamp/freeCodeCamp、github.com/vinta/awesome-python等10个热门仓库，每个类别访问100次的完整访问流量。
4. `github10_20250502.pkl`包含20250502采集的github.com/freeCodeCamp/freeCodeCamp、github.com/vinta/awesome-python等10个热门仓库，每个类别访问约100次的完整访问流量。（可配合0303做概念漂移验证）

5. `github100.pkl`包含采集的github.com/freeCodeCamp/freeCodeCamp、github.com/vinta/awesome-python等100个热门仓库，每个类别访问100次的完整访问流量。（可配合0303做开放世界验证）
6. `baidu10000.pkl`包含baidu、weibo等约10000个热门网站，每个类别访问1次的完整访问流量。（可配合0817做概念漂移验证）