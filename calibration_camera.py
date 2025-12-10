import cv2
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import os
import math
import json

class ImageRedactor: 

    def __init__(self):
        self.roi_polygon = []
        self.project_dir = os.path.dirname(__file__)

    def build_all_models(self, measurements):
        """
        Строим несколько моделей (poly, log, exp, rational, sqrt, power) и выбираем лучшую по AIC/BIC
        measurements: { (x0, y0): { (x2, y2): dist } }
        """
        # --- собираем данные ---
        y_pixels, distances_m = [], []
        for zero_pt, points_dict in measurements.items():
            for pt, dist in points_dict.items():
                y_pixels.append(pt[1])
                distances_m.append(dist)
        y_pixels = np.array(y_pixels)
        distances_m = np.array(distances_m)

        # --- список моделей: (название, функция, количество параметров) ---
        def poly2(y, a, b, c): return a*y**2 + b*y + c
        def poly3(y, a, b, c, d): return a*y**3 + b*y**2 + c*y + d
        def log1(y, a, b): return a*np.log(y) + b
        def sqrt_linear(y, a, b, c): return a*np.sqrt(y) + b*y + c
        def rational1(y, a, b, c): return (a*y + b) / (y + c)
        def rational2(y, a, b, c, d, e): return (a*y**2 + b*y + c) / (d*y + e)

        all_models = [
            ("poly2", poly2, 3),
            ("poly3", poly3, 4),
            ("log1", log1, 2),
            ("sqrt_linear", sqrt_linear, 3),
            ("rational1", rational1, 3),
            ("rational2", rational2, 5)
        ]

        results = {}
        plot_dir = os.path.join(self.project_dir, "model_plots")
        os.makedirs(plot_dir, exist_ok=True)

        n = len(y_pixels)
        for name, func, k in all_models:
            try:
                popt, _ = curve_fit(func, y_pixels, distances_m, maxfev=10000)
                residuals = distances_m - func(y_pixels, *popt)
                RSS = np.sum(residuals**2)
                AIC = 2*k + n * np.log(RSS/n)
                BIC = k*np.log(n) + n * np.log(RSS/n)

                results[name] = {
                    "func": func,
                    "params": popt,
                    "RSS": RSS,
                    "AIC": AIC,
                    "BIC": BIC
                }

                # --- график ---
                y_line = np.linspace(min(y_pixels), max(y_pixels), 200)
                plt.figure(figsize=(8,6))
                plt.scatter(y_pixels, distances_m, color='red', label='Измерения')
                plt.plot(y_line, func(y_line, *popt), label=f'{name} fit', color='blue')
                plt.xlabel('y-пиксели')
                plt.ylabel('Расстояние (м)')
                plt.title(f'Модель {name}: AIC={AIC:.2f}, BIC={BIC:.2f}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{name}.png"))
                plt.close()

            except Exception as e:
                print(f"Ошибка при подгонке модели {name}: {e}")

        # После выбора лучшей модели по AIC
        best_aic_model = min(results.items(), key=lambda x: x[1]['AIC'])
        best_bic_model = min(results.items(), key=lambda x: x[1]['BIC'])

        print(f"Лучшая модель по AIC: {best_aic_model[0]}, параметры: {best_aic_model[1]['params']}")
        print(f"Лучшая модель по BIC: {best_bic_model[0]}, параметры: {best_bic_model[1]['params']}")

        # --- функция предсказания для лучшей модели по AIC ---
        def predict(y_pixel):
            return best_aic_model[1]['func'](y_pixel, *best_aic_model[1]['params'])

        # --- ДОПОЛНИТЕЛЬНЫЙ график для лучшей модели ---
        y_line = np.linspace(min(y_pixels), max(y_pixels), 300)
        plt.figure(figsize=(10, 7))
        plt.scatter(y_pixels, distances_m, color='red', label='Измерения')
        plt.plot(y_line, predict(y_line), color='blue', linewidth=2, label=f'Лучшая модель ({best_aic_model[0]})')
        plt.xlabel('y-пиксели')
        plt.ylabel('Расстояние (м)')
        plt.title(f'Лучшая модель по AIC: {best_aic_model[0]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        best_plot_file = os.path.join(plot_dir, "best_model.png")
        plt.savefig(best_plot_file)
        plt.close()
        print(f"График лучшей модели сохранен в {best_plot_file}")

        return predict, results

    def find_closest_edge(self, polygon, click_point):
        min_dist = float("inf")
        closest_edge = None
        closest_edge_index = -1

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            # расстояние от точки до отрезка
            dist = self.point_to_segment_distance(click_point, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_edge = (p1, p2)
                closest_edge_index = i

        return closest_edge_index, closest_edge


    def point_to_segment_distance(self, p, a, b):
        # расстояние от точки p до отрезка ab
        px, py = p
        ax, ay = a
        bx, by = b

        dx = bx - ax
        dy = by - ay

        if dx == dy == 0:
            return math.hypot(px - ax, py - ay)

        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy

        return math.hypot(px - proj_x, py - proj_y)
    
    def select_edge(self, frame):
        print("\n=== ВЫБОР РЕБРА ===")
        print("Кликните на ребро ROI, которое хотите выбрать")
        print("Нажмите ENTER для подтверждения")

        clone = frame.copy()
        selected_edge_index = None
        hover_point = None

        def click_event(event, x, y, flags, param):
            nonlocal selected_edge_index, hover_point
            hover_point = (x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                idx, edge = self.find_closest_edge(self.roi_polygon, (x, y))
                selected_edge_index = idx
                print(f"Выбрано ребро: {idx} ({edge})")

            temp = clone.copy()
            cv2.polylines(temp, [self.roi_polygon], True, (0, 255, 0), 2)

            # Подсветить выбранное ребро
            if selected_edge_index is not None:
                p1 = tuple(self.roi_polygon[selected_edge_index])
                p2 = tuple(self.roi_polygon[(selected_edge_index + 1) % len(self.roi_polygon)])
                cv2.line(temp, p1, p2, (0, 0, 255), 4)

            cv2.imshow("Select Edge", temp)

        cv2.namedWindow("Select Edge")
        cv2.setMouseCallback("Select Edge", click_event)
        cv2.imshow("Select Edge", clone)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and selected_edge_index is not None:  # Enter
                break

        cv2.destroyWindow("Select Edge")
        return selected_edge_index

    def measure_points(self, frame, edge_index):
        print("\n=== ИЗМЕРЕНИЯ РАССТОЯНИЙ ===")
        print("1. Выберите точку на прямой (нулевую точку)")
        print("2. Затем выберите вторую точку в любом месте")
        print("3. Введите расстояние в метрах")
        print("4. Повторяйте измерения для этой же нулевой точки")
        print("5. Чтобы выбрать новую нулевую точку — снова кликните по прямой")
        print("6. ENTER — завершить")

        # Храним координату полной нулевой точки!
        # { (x0, y0): { (x2, y2): dist } }
        results = {}
        second_points = {}

        clone = frame.copy()

        p1 = tuple(self.roi_polygon[edge_index])
        p2 = tuple(self.roi_polygon[(edge_index + 1) % len(self.roi_polygon)])

        current_zero_point = None
        stage = 0

        def redraw_window():
            img = frame.copy()

            cv2.polylines(img, [self.roi_polygon], True, (0, 255, 0), 2)
            cv2.line(img, p1, p2, (0, 0, 255), 3)

            # Рисуем только реальные нулевые точки
            for zero_pt, points in second_points.items():
                cv2.circle(img, zero_pt, 7, (255, 255, 0), -1)

                for (pt, dist) in points:
                    cv2.circle(img, pt, 6, (0, 255, 255), -1)
                    # ЛИНИИ БОЛЬШЕ НЕ РИСУЕМ
                    cv2.putText(img, f"{dist}m", (pt[0] + 10, pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)

            # Текущая нулевая точка
            if current_zero_point is not None:
                cv2.circle(img, current_zero_point, 7, (0, 128, 255), -1)
                cv2.putText(img, "Zero Point",
                            (current_zero_point[0] + 10, current_zero_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

            cv2.imshow("Measure Points", img)

        def click_event(event, x, y, flags, param):
            nonlocal current_zero_point, stage

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            click_pt = (x, y)

            # Проверяем — попали ли по ребру
            is_on_edge = self.point_to_segment_distance(click_pt, p1, p2) < 6

            if is_on_edge:
                # Всегда создаём новую нулевую точку
                current_zero_point = click_pt
                results[current_zero_point] = {}
                second_points[current_zero_point] = []

                print(f"\nНовая нулевая точка: {current_zero_point}")
                stage = 1
                redraw_window()
                return

            # Иначе — это точка измерения
            if stage == 1:
                second_point = click_pt
                print(f"Выбрана вторая точка: {second_point}")

                dist_str = input("Введите расстояние (м): ").strip()
                try:
                    dist = float(dist_str)
                except:
                    print("Ошибка: введите число.")
                    return

                results[current_zero_point][second_point] = dist
                second_points[current_zero_point].append((second_point, dist))

                print(f"Добавлено: {second_point} → {dist} м")
                redraw_window()


        cv2.namedWindow("Measure Points")
        cv2.setMouseCallback("Measure Points", click_event)
        redraw_window()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 13:
                break

        cv2.destroyWindow("Measure Points")
        return results

    def setup_roi(self, frame):
        """
        Настройка области интереса (ROI) вручную с произвольным количеством точек
        """
        print("\n=== ВЫБОР ОБЛАСТИ ИНТЕРЕСА (ROI) ===")
        print("Инструкция:")
        print("1. ЛКМ - добавить точку")
        print("2. ПКМ - удалить последнюю точку") 
        print("3. 'r' - сбросить все точки")
        print("4. 'c' - завершить выбор (минимум 3 точки)")
        print("5. 'q' - отменить выбор ROI")
        print("6. ENTER - подтвердить выбор")

        clone = frame.copy()
        roi_points = []

        last_mouse_pos = None

        def click_event(event, x, y, flags, param):
            nonlocal last_mouse_pos
            last_mouse_pos = (x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
                print(f"Добавлена точка {len(roi_points)}: ({x}, {y})")

            elif event == cv2.EVENT_RBUTTONDOWN:
                if roi_points:
                    removed_point = roi_points.pop()
                    print(f"Удалена точка: {removed_point}")

            # === Новый функционал: вывод координат при движении мыши ===
            if event == cv2.EVENT_MOUSEMOVE:
                print(f"Курсор: ({x}, {y})", end="\r")  # перезапись одной строки

            # Перерисовка
            temp_clone = clone.copy()

            # Рисуем все точки
            for i, point in enumerate(roi_points):
                cv2.circle(temp_clone, point, 5, (0, 255, 0), -1)
                cv2.putText(temp_clone, str(i+1), (point[0]+10, point[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Линии между точками
            if len(roi_points) > 1:
                cv2.polylines(temp_clone, [np.array(roi_points, dtype=np.int32)],
                            isClosed=False, color=(0, 255, 0), thickness=2)

            # Закрытый полигон
            if len(roi_points) >= 3:
                closed = roi_points + [roi_points[0]]
                cv2.polylines(temp_clone, [np.array(closed, dtype=np.int32)],
                            isClosed=True, color=(255, 0, 0), thickness=2)

                overlay = temp_clone.copy()
                cv2.fillPoly(overlay, [np.array(closed, dtype=np.int32)], (0, 100, 0))
                cv2.addWeighted(overlay, 0.3, temp_clone, 0.7, 0, temp_clone)

            # === Рисуем текст с координатами возле курсора ===
            if last_mouse_pos is not None:
                mx, my = last_mouse_pos
                cv2.putText(temp_clone, f"({mx}, {my})",
                            (mx + 15, my - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)

                cv2.circle(temp_clone, (mx, my), 3, (0, 255, 255), -1)

            cv2.imshow("ROI Selection", temp_clone)


        cv2.namedWindow("ROI Selection")
        cv2.setMouseCallback("ROI Selection", click_event)
        
        # Первоначальное отображение
        cv2.imshow("ROI Selection", clone)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # сброс
                roi_points.clear()
                clone = frame.copy()
                cv2.imshow("ROI Selection", clone)
                print("Все точки сброшены.")
                
            elif key == ord('c'):  # завершить выбор
                if len(roi_points) >= 3:
                    break
                else:
                    print("Нужно как минимум 3 точки для создания области!")
                    
            elif key == 13:  # Enter - подтверждение
                if len(roi_points) >= 3:
                    break
                else:
                    print("Нужно как минимум 3 точки для создания области!")
                    
            elif key == ord('q'):  # выход
                cv2.destroyWindow("ROI Selection")
                print("Выбор ROI отменен.")
                return None

        cv2.destroyWindow("ROI Selection")

        # Замыкаем полигон
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
        
        print(f"\nROI установлен! Количество точек: {len(roi_points)}")
        print(f"Координаты: {self.roi_polygon.tolist()}")
        
        # Показываем финальный ROI на кадре
        final_display = frame.copy()
        cv2.polylines(final_display, [self.roi_polygon], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, point in enumerate(roi_points):
            cv2.circle(final_display, point, 6, (0, 0, 255), -1)
            cv2.putText(final_display, str(i+1), (point[0]+10, point[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Final ROI", final_display)
        cv2.waitKey(2000)  # Показываем 2 секунды
        cv2.destroyWindow("Final ROI")

        return self.roi_polygon
    
    def open_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Видео: {video_path}")
        print(f"Размер: {width}x{height}, FPS: {fps}, Всего кадров: {total_frames}")

        # Попробуем загрузить старый ROI и измерения
        config_file = os.path.join(self.project_dir, "roi_and_measurements.json")
        roi_polygon, edge_index, measurements = self.load_roi_and_measurements(config_file)

        # Чтение первого кадра
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось прочитать первый кадр")
            cap.release()
            return

        # --- Работа с ROI ---
        if roi_polygon is not None:
            use_old = input("Найден старый ROI. Использовать его? (y/n): ").strip().lower() == 'y'
            if use_old:
                self.roi_polygon = roi_polygon
                print("Используем старый ROI")
            else:
                roi_polygon = None

        if roi_polygon is None:
            # Создаём новый ROI
            user_roi = self.setup_roi(frame)
            edge_index = self.select_edge(frame)
            self.roi_polygon = user_roi

        # --- Работа с измерениями ---
        if measurements and edge_index is not None:
            use_old = input("Найдены старые измеренные точки. Использовать их? (y/n): ").strip().lower() == 'y'
            if use_old:
                print("Используем старые измеренные точки")
            else:
                measurements = self.measure_points(frame, edge_index)
        else:
            measurements = self.measure_points(frame, edge_index)

        # --- Сохраняем ROI и измерения ---
        self.save_roi_and_measurements(config_file, self.roi_polygon, edge_index, measurements)

        # --- Построение регрессионной модели ---
        # predict_func, params = self.build_exp3p_model(measurements)
        predict_func, all_results = self.build_all_models(measurements)

        # --- Финальный просмотр ROI и измерений ---
        ret, frame = cap.read()  # читаем первый кадр снова для финального просмотра
        if ret:
            final_frame = frame.copy()
            cv2.polylines(final_frame, [self.roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Рисуем точки измерений
            for zero_pt, points_dict in measurements.items():
                cv2.circle(final_frame, zero_pt, 7, (0, 128, 255), -1)
                cv2.putText(final_frame, "Zero", (zero_pt[0]+10, zero_pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
                for pt, dist in points_dict.items():
                    cv2.circle(final_frame, pt, 6, (0, 255, 255), -1)
                    cv2.putText(final_frame, f"{dist}m", (pt[0]+10, pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cursor_pos = [0, 0]

            def mouse_move(event, x, y, flags, param):
                nonlocal cursor_pos
                if event == cv2.EVENT_MOUSEMOVE:
                    cursor_pos = [x, y]

            cv2.namedWindow("Final View")
            cv2.setMouseCallback("Final View", mouse_move)

            while True:
                display_frame = final_frame.copy()
                # Показываем координаты курсора
                cv2.putText(display_frame, f"Cursor: ({cursor_pos[0]}, {cursor_pos[1]})",
                            (cursor_pos[0]+15, cursor_pos[1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Final View", display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 13:  # Enter
                    break

            cv2.destroyWindow("Final View")


        # # --- Основной цикл видео ---
        # frame_count = 0
        # car_count = 0

        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break

        #     # Визуализация ROI
        #     if self.roi_polygon is not None:
        #         cv2.polylines(frame, [self.roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        #         cv2.putText(frame, f'ROI ({len(self.roi_polygon)} точек)', 
        #                     tuple(self.roi_polygon[0]),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        #     cv2.imshow('Car Detection', frame)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('q'):
        #         break

        #     frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        # print(f"\nОбработка завершена! Найдено автомобилей: {car_count}")
        # return car_count

    
    def save_roi_and_measurements(self, filename, roi_polygon, edge_index, measurements):
        data = {
            "roi_polygon": roi_polygon.tolist() if isinstance(roi_polygon, np.ndarray) else roi_polygon,
            "edge_index": edge_index,
            "measurements": {
                str(zero_pt): {str(pt): dist for pt, dist in pts.items()}
                for zero_pt, pts in measurements.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"ROI и точки сохранены в {filename}")

    def load_roi_and_measurements(self, filename):
        if not os.path.exists(filename):
            return None, None, None
        with open(filename, 'r') as f:
            data = json.load(f)
        roi_polygon = np.array(data.get("roi_polygon", []), dtype=np.int32)
        edge_index = data.get("edge_index", None)
        measurements_raw = data.get("measurements", {})

        # Конвертируем строки обратно в кортежи
        measurements = {}
        for zero_pt_str, points_dict in measurements_raw.items():
            zero_pt = tuple(map(int, zero_pt_str.strip("()").split(",")))
            measurements[zero_pt] = {}
            for pt_str, dist in points_dict.items():
                pt = tuple(map(int, pt_str.strip("()").split(",")))
                measurements[zero_pt][pt] = dist

        return roi_polygon, edge_index, measurements

    
def main():
    imageRedactor = ImageRedactor()
    imageRedactor.open_video('./videos/20250517_042707_D1.mp4')

if __name__ == "__main__":
    main()