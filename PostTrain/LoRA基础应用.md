# LoRA基础应用

库的安装：torch、transformers、peft、dataset。其中，transformers和peft为huggingface的库

## 加载模型及数据

数据集较为重要，本次采用GPT-2及ELI5数据集的小部分，该集合包含问答对

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer,TrainingArguments,DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from dataset import load_dataset

# 加载模型及分词器
model_name = "gpt-2"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # 设置批处理的填充标记

#加载和准备数据集
dataset = load_dataset("eli5", split="train_asks[:1000]") # 使用小切片

# 基本预处理，对文本进行分词
def preprocess_function(examples):
    # 为因果语言模型训练链接问题及答案
    texts = [q + " " + a[0] for q, a in zip(examples['title'], examples['answer.text'])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_functions, batched=True,remove_columns=dataset.column_names)

print("基础模型已加载：", model_name)
print("数据集已加载并分词")
```

## 配置LoRA属性

```python
lora_config = LoraConfig(
	r = 16, # 秩
    lora_alpha = 32, # 缩放因子 alpha
    traget_modules = ["c_attn"], # LoRA应用在注意力机制的qkv投影
    lora_dropout = 0.5, # LoRA层Dropout概率 
    bias = "none", # 不设置偏执参数
    task_type = TaskType.CAUSAL_LM # 因果语言模型的训练任务
)

print("LoRA配置：\n", lora_config)
```

## 应用模型

```python
# 计算原始可训练参数
original_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(f"原始可训练参数为：{original_params}")

# 将LoRA配置应用于基础模型
lora_model = get_peft_model(base_model, lora_config)

# 计算LoRA可训练参数
lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"LoRA可训练参数为：{lora_params}")

# 计算减少量
reduction = (original_params - lora_params) / original_params * 100
print(f"参数减少：{reduction.2f}%")

# 打印可训练模块以供检验
lora_model.print_trainable_parameters()
```

## 设置训练循环

现在可以使用transformers.Trainer设置标准训练过程。主要区别在于传递的是peft修改后的参数，而不是base_model。Trainer将自动优化处理，只关注可训练的LoRA参数。

```python
# 定义训练参数
output_dir = "./lora_gpt2_eli5_results"
training_args = TrainingArguments(
	output_dir = output_dir,
    num_train_epochs = 1, # 为演示目的缩短周期
    per_device_train_batch_size = 4, # 根据GPU内存自行调整
    logging_steps = 50, 
    save_steps = 200,
    learning_rate = 2e-4, # LoRA的典型学习率
    fp16 = torch.cuda.is_avaliable(), # 如果可用，，使用混合精度
)

# 因果语言模型的数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 初始化训练器
trainer = Trainer(
	model = lora_model,
    args = training_args,
    train_dataset = tokenized_dataset,
    tokenizer = tokenizer,
    data_collator = data_collator
)

print("训练器已初始化，开始训练……")
```

## 训练模型

```python
# 开始训练
trainer.train()

print("训练完成。")
```

## 保存LoRA适配器

```python
# 定义保存适配器的路径
adapter_path = f"{output_dir}/final_adapter"

# 保存LoRA适配器权重
lora_model.save_pretrained(adapter_path)

print(f"LoRA适配器已保存到: {adapter_path}")

# 您可以验证保存的适配器目录的小尺寸
# !ls -lh {adapter_path}
```

保存的目录（`adapter_path`）将包含`adapter_model.bin`（LoRA权重）和`adapter_config.json`（使用的LoRA配置）等文件。其大小将以兆字节计，明显小于完整基础模型所需的千兆字节。

## 应用LoRA Adaptor

```python
# 再次加载基础模型（或使用内存中已有的模型）
base_model_reloaded = AutoModelForCausalLM.from_pretrained(model_name)

# 通过合并适配器权重加载PEFT模型
inference_model = PeftModel.from_pretrained(base_model_reloaded, adapter_path)

# 确保模型处于评估模式并在正确的设备上
inference_model.eval()
if torch.cuda.is_available():
    inference_model.to("cuda")

print("基础模型已加载并应用LoRA适配器进行推理。")

# 示例推理（可选）
prompt = "What is the main cause of climate change?"
inputs = tokenizer(prompt, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 生成文本
with torch.no_grad():
    outputs = inference_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- 示例生成 ---")
print("提示:", prompt)
print("生成内容:", generated_text)
print("------------------------")
```

上述代码中`outputs` 有 `**inputs`参数，这表示解包字典。

```python
# 定义函数
def my_function(name, age, city):
    print(f"{name} is {age} years old, lives in {city}")

# 参数字典
person = {'name': 'Alice', 'age': 25, 'city': 'Beijing'}

# 使用 ** 解包字典
my_function(**person)  # 等价于 my_function(name='Alice', age=25, city='Beijing')
```

如果是`*input`表示解包列表

```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
merged = [*list1, *list2]  # [1, 2, 3, 4, 5, 6]
print(merged)
```





