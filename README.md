# Модели

В обучении я использовал EfficientNet V2 (M) и ViT16/32 (b).

По какой-то причине добится качества >90 процентов на трансформерах не удалось (хотя многие достаточно просто получили такой результат с ViT), но EfficientNet показал себя хорошо.
Собственно сначала обучалась голова для классификации, затем постепенно размараживались остальные слои.
Данные, которые я отправил на проверку - это результаты переобученной модели.