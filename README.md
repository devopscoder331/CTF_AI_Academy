## CTF_AI_Academy

Для запуска CTF AI тасок нужно скачать ai-модельки в папку models, ниже пример:

```bash
# task 1
cd tasks/models
curl -vL https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/resolve/main/ggml-old-vic13b-q4_0.bin -O
```

**Таски:**

| Модуль  | Уязвимость      | Модель                   | Уровень | Порты          | Задание | Райтап |
| --------| ----------------| ------------------------ | ------- | ---------------| ------- | ------ |
| task 1  | Promt Injection | ggml-old-vic13b-q4_0.bin | Лёгкий  | 50011 <=> 5000 | ссылка  | ссылка |
