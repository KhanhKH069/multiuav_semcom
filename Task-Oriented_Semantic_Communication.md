**HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG**

**KHOA VIỄN THÔNG 1**

**ĐỀ CƯƠNG NGHIÊN CỨU KHOA HỌC SINH VIÊN**

**KHOA HỌC CÔNG NGHỆ SINH VIÊN 2026**

**ĐỀ TÀI**

**Task-oriented Semantic Communication cho hệ thống Multi-UAV giám sát và bám mục tiêu**

**Mã số: ……………………………**

|  |  |
| --- | --- |
| **Giảng viên hướng dẫn:** | **TS. Ngô Thị Thu Trang** |
| **Nhóm sinh viên thực hiện:** | **Nguyễn Vũ Kim Anh**  **Vũ Nam Khánh**  **Lê Hoàng Anh**  **Phan Quang Hiếu** |
| **Đơn vị:** | **Khoa Viễn thông 1 – PTIT** |

**Hà Nội – 2026**

**NHẬN XÉT, ĐÁNH GIÁ, CHO ĐIỂM**

**(CỦA GIÁO VIÊN PHẢN BIỆN)**

**Điểm:……………………………(Bằng chữ:………………………………..)**

|  |  |
| --- | --- |
|  | *Hà Nội, ngày tháng năm 2026*  **CÁN BỘ - GIẢNG VIÊN PHẢN BIỆN**  *(ký và ghi rõ họ tên)* |

**LỜI CẢM ƠN**

Trong suốt thời gian học tập và nghiên cứu tại Học viện Công nghệ Bưu chính Viễn thông, chúng em đã nhận được sự quan tâm, chỉ dạy tận tình của quý Thầy Cô trong Học viện nói chung và quý Thầy Cô Khoa Viễn thông 1 nói riêng. Những kiến thức được truyền đạt không chỉ là nền tảng cho quá trình nghiên cứu khoa học mà còn là hành trang quý báu để chúng em tự tin hơn trên con đường học tập và nghiên cứu.

Chúng em xin chân thành cảm ơn cô **TS. Ngô Thị Thu Trang** đã tận tâm hướng dẫn, định hướng và góp ý trong suốt quá trình xây dựng đề cương nghiên cứu này. Nhờ sự hướng dẫn của cô, nhóm đã từng bước hoàn thiện được ý tưởng và kế hoạch triển khai đề tài.

Do đây là đề tài mới được hình thành ở giai đoạn đề cương, kiến thức và kinh nghiệm nghiên cứu của nhóm còn nhiều hạn chế, chắc chắn không tránh khỏi thiếu sót. Nhóm rất mong nhận được sự góp ý từ quý Thầy Cô để đề cương và quá trình triển khai sau này được hoàn thiện hơn.

Chúng em xin chân thành cảm ơn!

**Nhóm sinh viên thực hiện đề tài**

**MỤC LỤC**

[MỞ ĐẦU](#mở-đầu)

[1. Tính cấp thiết của đề tài](#1-tính-cấp-thiết-của-đề-tài)

[2. Mục tiêu nghiên cứu](#2-mục-tiêu-nghiên-cứu)

[3. Đối tượng và phạm vi nghiên cứu](#3-đối-tượng-và-phạm-vi-nghiên-cứu)

[4. Phương pháp nghiên cứu](#4-phương-pháp-nghiên-cứu)

[5. Cấu trúc đề cương](#5-cấu-trúc-đề-cương)

[CHƯƠNG 1. TỔNG QUAN VỀ SEMANTIC COMMUNICATION VÀ BÀI TOÁN MULTI-UAV TRACKING](#chương-1-tổng-quan-về-semantic-communication-và-bài-toán-multi-uav-tracking)

[1.1 Bối cảnh nghiên cứu và tính cấp thiết](#11-bối-cảnh-nghiên-cứu-và-tính-cấp-thiết)

[1.2 Tổng quan tài liệu](#12-tổng-quan-tài-liệu)

[1.2.1 Semantic Communication và Task-oriented Compression](#121-semantic-communication-và-task-oriented-compression)

[1.2.2 Bộ dữ liệu UAV123 cho Visual Tracking](#122-bộ-dữ-liệu-uav123-cho-visual-tracking)

[1.2.3 Các phương pháp Fusion đa cảm biến](#123-các-phương-pháp-fusion-đa-cảm-biến)

[1.3 Đối tượng và phạm vi nghiên cứu chi tiết](#13-đối-tượng-và-phạm-vi-nghiên-cứu-chi-tiết)

[1.4 Kết luận chương 1](#14-kết-luận-chương-1)

[CHƯƠNG 2. THIẾT KẾ KIẾN TRÚC HỆ THỐNG](#chương-2-thiết-kế-kiến-trúc-hệ-thống)

[2.1 Tổng quan kiến trúc hệ thống](#21-tổng-quan-kiến-trúc-hệ-thống)

[2.2 Encoder task-oriented (ResNet18)](#22-encoder-task-oriented-resnet18)

[2.3 Giao thức trao đổi message](#23-giao-thức-trao-đổi-message)

[2.4 Fusion đa UAV](#24-fusion-đa-uav)

[2.5 Môi trường mô phỏng tracking mục tiêu di động](#25-môi-trường-mô-phỏng-tracking-mục-tiêu-di-động)

[2.6 PoC phần cứng: 3× Tello + RC car](#26-poc-phần-cứng-3-tello--rc-car)

[2.7 Kết luận chương 2](#27-kết-luận-chương-2)

[CHƯƠNG 3. KẾ HOẠCH TRIỂN KHAI, KẾT QUẢ DỰ KIẾN VÀ RỦI RO](#chương-3-kế-hoạch-triển-khai-kết-quả-dự-kiến-và-rủi-ro)

[3.1 Kế hoạch thực hiện theo giai đoạn](#31-kế-hoạch-thực-hiện-theo-giai-đoạn)

[3.2 Phân công nhiệm vụ](#32-phân-công-nhiệm-vụ)

[3.3 Kết quả dự kiến](#33-kết-quả-dự-kiến)

[3.4 Rủi ro và phương án dự phòng](#34-rủi-ro-và-phương-án-dự-phòng)

[3.5 Định hướng công bố khoa học](#35-định-hướng-công-bố-khoa-học)

[3.6 Kết luận chương 3](#36-kết-luận-chương-3)

[KẾT LUẬN](#kết-luận)

[TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)

#

**DANH MỤC HÌNH VẼ (DỰ KIẾN)**

*Ghi chú: Đề tài đang ở giai đoạn đề cương, các hình vẽ dưới đây sẽ được xây dựng và bổ sung trong quá trình triển khai.*

Hình 2.1. Kiến trúc tổng thể hệ thống Multi-UAV Semantic Communication

Hình 2.2. Pipeline trích xuất đặc trưng của Encoder ResNet18

Hình 2.3. Cấu trúc message ngữ nghĩa trao đổi giữa UAV và trạm fusion

Hình 2.4. Sơ đồ khối Fusion đa UAV (Kalman Filter)

Hình 2.5. Sơ đồ bố trí PoC: 3× Tello + RC car trong không gian 6×6 m

**DANH MỤC BẢNG BIỂU**

Bảng 3.1. Kế hoạch thực hiện theo giai đoạn

Bảng 3.2. Phân công nhiệm vụ

Bảng 3.3. Rủi ro và phương án dự phòng

#

|  |  |  |
| --- | --- | --- |
| **THUẬT NGỮ VIẾT TẮT** |  |  |
| **Viết tắt** | **Tiếng Anh** | **Tiếng Việt** |
| UAV | Unmanned Aerial Vehicle | Phương tiện bay không người lái |
| RC car | Remote Control car | Xe điều khiển từ xa |
| PoC | Proof of Concept | Bằng chứng khái niệm |
| KF | Kalman Filter | Bộ lọc Kalman |
| LSTM | Long Short-Term Memory | Mạng bộ nhớ ngắn-dài hạn |
| ResNet | Residual Network | Mạng nơ-ron phần dư |
| JSCC | Joint Source-Channel Coding | Mã hóa nguồn-kênh kết hợp |
| UDP | User Datagram Protocol | Giao thức gói dữ liệu người dùng |
| TCP | Transmission Control Protocol | Giao thức điều khiển truyền vận |
| FPS | Frames Per Second | Khung hình mỗi giây |
| SDK | Software Development Kit | Bộ công cụ phát triển phần mềm |
| Wi-Fi | Wireless Fidelity | Mạng không dây |
| ID | Identifier | Định danh |

#

# MỞ ĐẦU

## 1. Tính cấp thiết của đề tài

Các hệ thống multi-UAV hợp tác giám sát và bám mục tiêu ngày càng phổ biến trong các ứng dụng an ninh, cứu hộ và giao thông thông minh. Mô hình truyền thống truyền toàn bộ luồng video thô từ UAV về trạm điều khiển (thường 25–50 Mbps mỗi UAV) đòi hỏi băng thông lớn, độ trễ cao và không khả thi khi số lượng UAV tăng lên hoặc khi hoạt động trong môi trường mạng hạn chế (vùng thiên tai, khu vực không có hạ tầng viễn thông ổn định).

Semantic Communication (truyền thông theo ngữ nghĩa) là hướng nghiên cứu mới nổi, trong đó thiết bị chỉ trích xuất và truyền đi phần thông tin thực sự cần thiết cho tác vụ (task-relevant information) thay vì toàn bộ dữ liệu thô. Áp dụng nguyên lý này cho bài toán multi-UAV tracking, mỗi UAV chỉ cần gửi về vị trí và đặc trưng (feature) của mục tiêu — một vector vài trăm byte — thay vì cả luồng video, giúp giảm băng thông tới hơn 3 bậc độ lớn trong khi vẫn duy trì được chất lượng bám mục tiêu nhờ phối hợp (fusion) thông tin từ nhiều UAV.

Xuất phát từ những hạn chế của mô hình truyền video thô truyền thống, đề tài "Task-oriented Semantic Communication cho hệ thống Multi-UAV giám sát và bám mục tiêu" được thực hiện nhằm xây dựng và kiểm chứng một hệ thống truyền thông theo hướng tác vụ, đánh giá định lượng mức giảm băng thông và độ chính xác bám mục tiêu, đồng thời triển khai một PoC (Proof of Concept) trên phần cứng thực để minh chứng tính khả thi.

## 2. Mục tiêu nghiên cứu

**Mục tiêu tổng quát:** Xây dựng một hệ thống truyền thông theo hướng tác vụ (task-oriented) cho phép nhiều UAV phối hợp bám mục tiêu di động bằng cách chỉ trao đổi thông tin ngữ nghĩa cô đọng (vị trí, đặc trưng ngoại hình mục tiêu) thay vì luồng video thô, từ đó giảm mạnh yêu cầu băng thông trong khi vẫn đảm bảo độ chính xác bám mục tiêu và có lợi ích phối hợp rõ rệt so với một UAV đơn lẻ.

**Mục tiêu cụ thể:**

* Xây dựng khối encoder task-oriented dựa trên ResNet18, trích xuất từ mỗi khung hình một vector đặc trưng cố định 132 phần tử (bounding box + appearance embedding) đại diện cho mục tiêu quan sát được.
* Thiết kế giao thức trao đổi message nhẹ (~200 byte/UAV) giữa các UAV và trạm fusion qua mạng UDP/TCP thông thường.
* Xây dựng khối fusion đa UAV (Kalman Filter làm nền tảng chính, có so sánh với phương án Attention + LSTM) để hợp nhất thông tin từ nhiều UAV thành quỹ đạo mục tiêu thống nhất.
* Xây dựng môi trường mô phỏng (simulation) tracking mục tiêu di động để đánh giá thuật toán trước khi triển khai phần cứng.
* Triển khai PoC thực nghiệm với 3× DJI Tello và một xe điều khiển từ xa (RC car) đóng vai trò mục tiêu, trong không gian thử nghiệm 6×6 m.
* Đánh giá định lượng: băng thông tiêu thụ mỗi UAV, độ chính xác bám mục tiêu (tracking accuracy), và mức tăng hiệu năng (gain) khi phối hợp nhiều UAV so với một UAV đơn lẻ.

## 3. Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu:** Hệ thống truyền thông và xử lý thông tin giữa nhiều UAV trong bài toán bám mục tiêu di động hợp tác, bao gồm: mô hình trích xuất đặc trưng ngữ nghĩa từ ảnh (encoder), giao thức trao đổi message, và thuật toán hợp nhất thông tin đa nguồn (fusion).

**Phạm vi nghiên cứu:** Để đảm bảo tính khả thi trong khuôn khổ đề tài NCKH sinh viên, nhóm giới hạn phạm vi như sau:

* Trong phạm vi: xử lý ảnh/tracking trên edge (laptop/PC nối trực tiếp với luồng video Tello), thiết kế và đánh giá định dạng message ngữ nghĩa cô đọng, thuật toán fusion (Kalman Filter, có mở rộng so sánh LSTM), mô phỏng và PoC phần cứng quy mô nhỏ.
* Ngoài phạm vi: không thực hiện Joint Source-Channel Coding (JSCC) và không mô phỏng kênh vô tuyến vật lý (fading, nhiễu đa đường). Việc truyền message giữa các UAV và trạm điều khiển sử dụng mạng Wi-Fi với giao thức UDP/TCP tiêu chuẩn, xem kênh truyền là gần như tin cậy ở khoảng cách thử nghiệm PoC (trong bán kính vài chục mét).

Việc thu hẹp phạm vi này giúp nhóm tập trung nguồn lực vào phần đóng góp cốt lõi — chứng minh định lượng lợi ích của việc nén dữ liệu theo hướng tác vụ (task-oriented compression) — thay vì dàn trải sang lý thuyết truyền thông tầng vật lý, vốn đòi hỏi khối lượng công việc vượt quy mô một đề tài sinh viên trong một học kỳ.

## 4. Phương pháp nghiên cứu

* Phương pháp nghiên cứu tài liệu: khảo sát các công trình về semantic communication, task-oriented compression, và các thuật toán fusion đa cảm biến (Kalman Filter, Attention-based fusion).
* Phương pháp thực nghiệm: huấn luyện và đánh giá encoder trên tập dữ liệu UAV123; xây dựng bộ dữ liệu multi-view riêng từ mô phỏng và từ chính hệ thống PoC để kiểm thử khối fusion.
* Phương pháp mô phỏng: xây dựng môi trường sim tracking mục tiêu di động trước khi triển khai phần cứng, nhằm kiểm chứng thuật toán fusion với chi phí thấp và có thể lặp lại thí nghiệm nhiều lần.
* Phương pháp đo lường – đánh giá định lượng: so sánh băng thông, độ trễ và độ chính xác giữa phương án baseline (truyền video thô) và phương án đề xuất (truyền message ngữ nghĩa).

## 5. Cấu trúc đề cương

Nội dung đề cương gồm 3 chương chính:

* Chương 1: Tổng quan về Semantic Communication và bài toán Multi-UAV Tracking – trình bày bối cảnh, khảo sát tài liệu liên quan và phạm vi nghiên cứu chi tiết.
* Chương 2: Thiết kế kiến trúc hệ thống – trình bày kiến trúc tổng thể, khối encoder, giao thức message, khối fusion, môi trường mô phỏng và thiết kế PoC phần cứng.
* Chương 3: Kế hoạch triển khai, kết quả dự kiến và rủi ro – trình bày kế hoạch theo giai đoạn, phân công nhiệm vụ, kết quả kỳ vọng, rủi ro và định hướng công bố khoa học.

# CHƯƠNG 1. TỔNG QUAN VỀ SEMANTIC COMMUNICATION VÀ BÀI TOÁN MULTI-UAV TRACKING

## 1.1 Bối cảnh nghiên cứu và tính cấp thiết

Các hệ thống multi-UAV hợp tác giám sát và bám mục tiêu ngày càng phổ biến trong các ứng dụng an ninh, cứu hộ và giao thông thông minh. Mô hình truyền thống truyền toàn bộ luồng video thô từ UAV về trạm điều khiển (thường 25–50 Mbps mỗi UAV) đòi hỏi băng thông lớn, độ trễ cao và không khả thi khi số lượng UAV tăng lên hoặc khi hoạt động trong môi trường mạng hạn chế (vùng thiên tai, khu vực không có hạ tầng viễn thông ổn định). Đây là động lực chính thúc đẩy nhóm hướng tới một giải pháp truyền thông tiết kiệm băng thông hơn nhưng vẫn đảm bảo chất lượng tác vụ bám mục tiêu.

## 1.2 Tổng quan tài liệu

### 1.2.1 Semantic Communication và Task-oriented Compression

Semantic Communication (truyền thông theo ngữ nghĩa) là hướng nghiên cứu mới nổi, trong đó thiết bị chỉ trích xuất và truyền đi phần thông tin thực sự cần thiết cho tác vụ (task-relevant information) thay vì toàn bộ dữ liệu thô. Áp dụng nguyên lý này cho bài toán multi-UAV tracking, mỗi UAV chỉ cần gửi về vị trí và đặc trưng (feature) của mục tiêu — một vector vài trăm byte — thay vì cả luồng video, giúp giảm băng thông tới hơn 3 bậc độ lớn trong khi vẫn duy trì được chất lượng bám mục tiêu nhờ phối hợp (fusion) thông tin từ nhiều UAV.

### 1.2.2 Bộ dữ liệu UAV123 cho Visual Tracking

UAV123 là bộ dữ liệu chuẩn cho bài toán single-object visual tracking từ góc nhìn UAV, gồm hơn 100 video với nhãn bounding box ground-truth theo từng khung hình. Trong đề tài này, UAV123 được sử dụng để huấn luyện và đánh giá khối encoder (trích xuất bbox + appearance embedding) ở mức một UAV. Vì UAV123 không có kịch bản multi-UAV/multi-view sẵn có, bộ dữ liệu phục vụ đánh giá khối fusion đa UAV sẽ được nhóm tự xây dựng, kết hợp dữ liệu mô phỏng nhiều góc nhìn và dữ liệu thu thập trực tiếp từ hệ thống PoC 3× Tello. Việc phân định rõ ràng ranh giới sử dụng UAV123 (chỉ ở mức encoder) cần được nêu minh bạch trong báo cáo để tránh gây hiểu nhầm về phạm vi của bộ dữ liệu chuẩn này.

### 1.2.3 Các phương pháp Fusion đa cảm biến

Kalman Filter (và biến thể Extended Kalman Filter) là phương pháp lọc truyền thống, ổn định và ít phụ thuộc vào lượng dữ liệu huấn luyện, phù hợp làm nền tảng chính cho khối fusion đa UAV. Bên cạnh đó, các phương pháp học máy như Attention kết hợp LSTM cho phép mô hình hóa quan hệ phi tuyến phức tạp hơn giữa các quan sát, nhưng đòi hỏi dữ liệu huấn luyện lớn hơn. Việc so sánh hai hướng tiếp cận này (một hướng lọc cổ điển, một hướng học sâu) là cơ sở để nhóm lựa chọn phương án phù hợp nhất với quy mô dữ liệu thực tế của đề tài.

## 1.3 Đối tượng và phạm vi nghiên cứu chi tiết

**Đối tượng nghiên cứu:** Hệ thống truyền thông và xử lý thông tin giữa nhiều UAV trong bài toán bám mục tiêu di động hợp tác, bao gồm: mô hình trích xuất đặc trưng ngữ nghĩa từ ảnh (encoder), giao thức trao đổi message, và thuật toán hợp nhất thông tin đa nguồn (fusion).

**Phạm vi nghiên cứu:**

* Trong phạm vi: xử lý ảnh/tracking trên edge (laptop/PC nối trực tiếp với luồng video Tello), thiết kế và đánh giá định dạng message ngữ nghĩa cô đọng, thuật toán fusion (Kalman Filter, có mở rộng so sánh LSTM), mô phỏng và PoC phần cứng quy mô nhỏ.
* Ngoài phạm vi: không thực hiện Joint Source-Channel Coding (JSCC) và không mô phỏng kênh vô tuyến vật lý (fading, nhiễu đa đường). Việc truyền message giữa các UAV và trạm điều khiển sử dụng mạng Wi-Fi với giao thức UDP/TCP tiêu chuẩn, xem kênh truyền là gần như tin cậy ở khoảng cách thử nghiệm PoC (trong bán kính vài chục mét).

Việc thu hẹp phạm vi này giúp nhóm tập trung nguồn lực vào phần đóng góp cốt lõi — chứng minh định lượng lợi ích của việc nén dữ liệu theo hướng tác vụ (task-oriented compression) — thay vì dàn trải sang lý thuyết truyền thông tầng vật lý, vốn đòi hỏi khối lượng công việc vượt quy mô một đề tài sinh viên trong một học kỳ.

## 1.4 Kết luận chương 1

Chương 1 đã trình bày bối cảnh và tính cấp thiết của đề tài, khảo sát các hướng tài liệu liên quan gồm semantic communication/task-oriented compression, bộ dữ liệu chuẩn UAV123 và các phương pháp fusion đa cảm biến, đồng thời xác định rõ đối tượng và phạm vi nghiên cứu. Đây là cơ sở để xây dựng kiến trúc hệ thống chi tiết trong chương 2.

# CHƯƠNG 2. THIẾT KẾ KIẾN TRÚC HỆ THỐNG

## 2.1 Tổng quan kiến trúc hệ thống

Hệ thống được thiết kế theo nguyên lý task-oriented semantic communication: mỗi UAV chạy một khối encoder cục bộ để trích xuất thông tin ngữ nghĩa cô đọng từ khung hình quan sát được, sau đó chỉ gửi một message nhẹ (không phải video) về trạm fusion trung tâm. Trạm fusion hợp nhất thông tin từ nhiều UAV thành một quỹ đạo mục tiêu thống nhất, có độ chính xác cao hơn so với dùng một UAV đơn lẻ nhờ tận dụng nhiều góc nhìn.

Kiến trúc gồm bốn khối chính, sẽ được trình bày lần lượt trong các mục 2.2–2.5, và một hệ thống PoC phần cứng minh chứng tính khả thi được trình bày ở mục 2.6:

* Khối Encoder task-oriented (ResNet18) chạy trên từng UAV/edge.
* Giao thức trao đổi message ngữ nghĩa cô đọng (~200 byte/UAV).
* Khối Fusion đa UAV (Kalman Filter, có so sánh với Attention + LSTM).
* Môi trường mô phỏng phục vụ kiểm chứng thuật toán trước khi triển khai phần cứng.

![Hình 2.1. Kiến trúc tổng thể hệ thống Multi-UAV Semantic Communication](images/architecture.png)


## 2.2 Encoder task-oriented (ResNet18)

Khối encoder sử dụng backbone ResNet18 (có thể tận dụng trọng số pretrained trên ImageNet rồi fine-tune trên UAV123) để trích xuất từ mỗi khung hình một vector đặc trưng cố định gồm 132 phần tử, dự kiến chia thành:

* 4 giá trị bounding box chuẩn hoá (x, y, w, h) xác định vị trí và kích thước mục tiêu trong khung hình.
* 128 giá trị appearance embedding (đặc trưng ngoại hình) phục vụ việc đối chiếu/liên kết mục tiêu giữa các UAV (re-identification) trong bước fusion.

Mô hình được huấn luyện bằng tập dữ liệu UAV123 thông qua hàm Loss tổng hợp: kết hợp MSE (Mean Squared Error) cho phần dự đoán Bounding Box và Triplet Loss (hoặc Cosine Embedding Loss) để tối ưu hóa không gian vector của Appearance Embedding, đảm bảo khả năng phân biệt mục tiêu tốt nhất. Yêu cầu kỹ thuật: mô hình cần chạy thời gian thực (real-time) trên phần cứng edge (laptop/PC), đảm bảo tốc độ xử lý đáp ứng tần suất khung hình thực nghiệm (mục tiêu ≥10–15 FPS).

## 2.3 Giao thức trao đổi message

Mỗi UAV, sau khi trích xuất vector 132 phần tử, đóng gói thành message có kích thước xấp xỉ 200 byte (bao gồm ID UAV, timestamp, 132 giá trị float lượng tử hoá, checksum) và gửi định kỳ về trạm fusion qua UDP trên mạng Wi-Fi cục bộ. Thiết kế message hướng tới tối giản overhead trong khi vẫn đảm bảo đồng bộ thời gian tương đối giữa các UAV để phục vụ bước hợp nhất dữ liệu.

## 2.4 Fusion đa UAV

Khối fusion chính sử dụng Kalman Filter (hoặc Extended Kalman Filter nếu quỹ đạo mục tiêu có thành phần phi tuyến) để hợp nhất các quan sát vị trí từ nhiều UAV thành một ước lượng quỹ đạo thống nhất. Đặc biệt, hệ thống tích hợp cơ chế **Liên kết dữ liệu (Data Association)**: tính toán Cosine Similarity của vector Appearance Embedding từ quan sát mới so với mục tiêu đang theo dõi. Các quan sát có độ tương đồng thấp (dưới ngưỡng 0.7) sẽ bị bộ lọc từ chối nhằm loại bỏ nhiễu (distractors). Đây là phương án khả thi, ổn định và khai thác tối đa lợi thế của Semantic Communication.

Song song, nhóm triển khai một phương án so sánh dựa trên Attention + LSTM quy mô nhỏ, huấn luyện trên dữ liệu quỹ đạo mô phỏng, nhằm đánh giá đối chiếu hiệu năng giữa phương pháp học máy và phương pháp lọc truyền thống. Kết quả so sánh (độ chính xác, độ trễ xử lý, yêu cầu dữ liệu huấn luyện) sẽ là một đóng góp học thuật bổ sung cho báo cáo.

## 2.5 Môi trường mô phỏng tracking mục tiêu di động

Trước khi triển khai phần cứng, nhóm xây dựng môi trường mô phỏng gồm nhiều UAV ảo quan sát một mục tiêu di động theo quỹ đạo có kịch bản (thẳng, cong, đổi hướng đột ngột) để kiểm chứng thuật toán fusion, đo độ chính xác bám mục tiêu và tinh chỉnh tham số trước khi thử nghiệm thực. Môi trường này cũng tích hợp **mô phỏng rớt mạng ngẫu nhiên (UDP Packet Loss)** với xác suất điều chỉnh được (ví dụ: 15%). Thông qua đó, hệ thống có thể chứng minh định lượng khả năng bù đắp thông tin và tính bền vững (robustness) vượt trội của mạng Multi-UAV so với UAV đơn lẻ khi hoạt động trên kênh truyền chập chờn.

## 2.6 PoC phần cứng: 3× Tello + RC car

Hệ thống PoC gồm 3 UAV DJI Tello (khuyến nghị bản Tello EDU để hỗ trợ ổn định việc kết nối đồng thời nhiều thiết bị) quan sát một xe điều khiển từ xa (RC car) đóng vai trò mục tiêu di động, hoạt động trong không gian thử nghiệm 6×6 m. Ba UAV được bố trí ở độ cao và góc nhìn khác nhau để tối đa hoá lợi ích của việc phối hợp đa góc nhìn. Dashboard giám sát hiển thị: trạng thái kết nối của 3 Tello, nhật ký các message ngữ nghĩa được gửi về (thay vì luồng video), và quỹ đạo mục tiêu theo thời gian thực sau khi fusion.

## 2.7 Kết luận chương 2

Chương 2 đã trình bày kiến trúc hệ thống task-oriented semantic communication đề xuất cho bài toán multi-UAV tracking, gồm khối encoder ResNet18, giao thức message ngữ nghĩa cô đọng, khối fusion đa UAV (Kalman Filter, có đối chiếu với Attention + LSTM), môi trường mô phỏng và thiết kế PoC phần cứng 3× Tello + RC car. Kiến trúc này là cơ sở để triển khai và đánh giá định lượng trong chương 3.

# CHƯƠNG 3. KẾ HOẠCH TRIỂN KHAI, KẾT QUẢ DỰ KIẾN VÀ RỦI RO

## 3.1 Kế hoạch thực hiện theo giai đoạn

**Bảng 3.1. Kế hoạch thực hiện theo giai đoạn**

| **Giai đoạn** | **Nội dung công việc** | **Kết quả dự kiến** |
| --- | --- | --- |
| Tuần 1–2 | Khảo sát tài liệu, chốt kiến trúc hệ thống, chuẩn bị dữ liệu UAV123, cài đặt môi trường phát triển và SDK Tello | Đề cương hoàn chỉnh, kiến trúc hệ thống, môi trường dev sẵn sàng |
| Tuần 3–5 | Xây dựng và huấn luyện encoder ResNet18 trên UAV123; thiết kế định dạng message 200 byte | Encoder chạy real-time, xuất đúng 132 floats; đặc tả giao thức message |
| Tuần 6–8 | Cài đặt Kalman Filter fusion; xây dựng môi trường mô phỏng tracking đa UAV; thử nghiệm so sánh với LSTM | Kết quả mô phỏng, bảng so sánh KF vs LSTM |
| Tuần 9–11 | Triển khai PoC: kết nối đồng thời 3× Tello, tích hợp encoder + network + fusion, xây dựng dashboard | Hệ thống PoC hoạt động ổn định trong không gian 6×6 m |
| Tuần 12–13 | Đo lường định lượng (băng thông, accuracy, gain phối hợp), quay demo, viết báo cáo | Số liệu đánh giá đầy đủ, video demo, báo cáo NCKH hoàn chỉnh |

## 3.2 Phân công nhiệm vụ

**Bảng 3.2. Phân công nhiệm vụ**

| **Nhóm** | **Thành viên** | **Nhiệm vụ chính** |
| --- | --- | --- |
| AI & Edge | Kim Anh (+ 1 thành viên hỗ trợ) | Xây dựng, tối ưu encoder ResNet18; đảm bảo trích xuất đúng 132 floats; tối ưu xử lý đa luồng (multi-threading) để tránh nghẽn khi nhận dữ liệu |
| Control & Network | Quang Hiếu (+ 1 thành viên hỗ trợ) | Lập trình điều khiển bay 3× Tello qua SDK; thiết kế và kiểm thử giao thức message UDP; đảm bảo ổn định kết nối Wi-Fi khi bay đồng thời |
| Data & Fusion | Hoàng Anh | Thu thập/xử lý dữ liệu UAV123 và dữ liệu mô phỏng; cài đặt Kalman Filter và LSTM fusion; xây dựng biểu đồ so sánh băng thông cho báo cáo |
| Dashboard & Báo cáo | Nam Khánh | Xây dựng dashboard giám sát (trạng thái kết nối, log message, trực quan hoá quỹ đạo); tổng hợp số liệu và soạn thảo báo cáo NCKH |

*Ghi chú: danh sách thành viên phụ trách từng nhóm nhiệm vụ cần được đối chiếu với danh sách "Nhóm sinh viên thực hiện" ở trang bìa để đảm bảo thống nhất tên gọi.*

## 3.3 Kết quả dự kiến

* Băng thông tiêu thụ mỗi UAV giảm xuống dưới 1 kbps, so với khoảng 50 Mbps của phương án truyền video thô đã nén (H.264) — mức giảm hơn 3 bậc độ lớn.
* Độ chính xác bám mục tiêu (tracking accuracy) đạt trên 85% trong không gian thử nghiệm 6×6 m.
* Mức tăng hiệu năng (gain) khi phối hợp 3 UAV ước tính đạt khoảng +15% so với phương án dùng một UAV đơn lẻ.
* Bảng so sánh định lượng giữa phương án Kalman Filter và Attention + LSTM cho bài toán fusion.
* Hệ thống PoC hoạt động thực tế, có video minh chứng, làm cơ sở cho báo cáo NCKH và khả năng mở rộng thành bài báo khoa học.

## 3.4 Rủi ro và phương án dự phòng

**Bảng 3.3. Rủi ro và phương án dự phòng**

| **Rủi ro** | **Ảnh hưởng** | **Phương án dự phòng** |
| --- | --- | --- |
| 3× Tello bay đồng thời bị nhiễu chéo Wi-Fi, mất kết nối | PoC không chạy ổn định, không quay được demo | Dùng Tello EDU, phân kênh Wi-Fi riêng cho từng UAV; kiểm thử kết nối đồng thời sớm ngay từ tuần 3–4 |
| LSTM không hội tụ do thiếu dữ liệu quỹ đạo huấn luyện | Thiếu kết quả so sánh fusion | Kalman Filter là phương án chính, đủ để chạy toàn bộ hệ thống; LSTM chỉ là thử nghiệm bổ sung, không nằm trên đường găng |
| Không gian 6×6 m quá nhỏ cho 3 UAV + RC car cùng hoạt động | Rủi ro va chạm, dữ liệu quan sát bị trùng lặp góc nhìn | Phân tầng độ cao bay khác nhau cho từng UAV; giới hạn tốc độ RC car trong thử nghiệm |

## 3.5 Định hướng công bố khoa học

Với quy mô một PoC UDP/Wi-Fi không mô phỏng kênh vật lý, đề tài phù hợp hơn với các venue tập trung vào ứng dụng và hệ thống (systems/applications) thay vì các tạp chí lý thuyết truyền thông thuần tuý. Đề xuất định hướng công bố:

* Kỷ yếu hội nghị trong nước / hội nghị NCKH sinh viên PTIT làm bước công bố đầu tiên.
* Các venue quốc tế phù hợp về ứng dụng semantic communication / edge AI cho UAV: IEEE Internet of Things Journal, IEEE Sensors Journal, hoặc các workshop chuyên đề Semantic Communication tại IEEE ICC/ISIT.
* IEEE Transactions on Communications chỉ khả thi nếu đề tài được mở rộng thêm thành phần lý thuyết truyền thông (mô hình hoá kênh, rate-distortion, JSCC) ở giai đoạn nghiên cứu tiếp theo — nằm ngoài phạm vi phiên bản NCKH sinh viên hiện tại.

## 3.6 Kết luận chương 3

Chương 3 đã trình bày kế hoạch thực hiện theo 5 giai đoạn trong 13 tuần, phân công nhiệm vụ cụ thể cho 4 nhóm chuyên trách (AI & Edge, Control & Network, Data & Fusion, Dashboard & Báo cáo), các kết quả kỳ vọng đạt được về băng thông, độ chính xác bám mục tiêu và mức tăng hiệu năng phối hợp đa UAV, cùng các rủi ro chính và phương án dự phòng tương ứng. Định hướng công bố khoa học cũng được xác định rõ, phù hợp với quy mô và phạm vi của đề tài NCKH sinh viên.

# KẾT LUẬN

Đề cương đã xác định rõ tính cấp thiết, mục tiêu, đối tượng, phạm vi và phương pháp nghiên cứu cho đề tài xây dựng hệ thống task-oriented semantic communication cho multi-UAV giám sát và bám mục tiêu. Điểm cốt lõi của đề tài là chứng minh định lượng lợi ích của việc nén dữ liệu theo hướng tác vụ: thay vì truyền toàn bộ luồng video thô (25–50 Mbps/UAV), mỗi UAV chỉ trích xuất và gửi đi một vector đặc trưng ngữ nghĩa cô đọng (~200 byte/UAV), giúp giảm băng thông hơn 3 bậc độ lớn trong khi vẫn duy trì được chất lượng bám mục tiêu nhờ cơ chế fusion đa UAV.

Kiến trúc đề xuất gồm bốn thành phần chính — encoder task-oriented dựa trên ResNet18, giao thức message nhẹ, khối fusion đa UAV (Kalman Filter làm nền tảng, có đối chiếu với Attention + LSTM) và môi trường mô phỏng kiểm chứng — sẽ được hiện thực hoá qua một PoC thực nghiệm với 3× DJI Tello và một RC car mục tiêu trong không gian 6×6 m. Kế hoạch triển khai 13 tuần, phân công nhiệm vụ theo 4 nhóm chuyên trách, cùng các phương án dự phòng cho những rủi ro kỹ thuật chính (nhiễu Wi-Fi, hội tụ LSTM, không gian thử nghiệm hạn chế) đã được xây dựng nhằm đảm bảo tính khả thi của đề tài trong một học kỳ.

Kết quả dự kiến — băng thông dưới 1 kbps/UAV, tracking accuracy trên 85%, gain phối hợp khoảng +15% — nếu đạt được sẽ là cơ sở định lượng vững chắc để mở rộng đề tài thành bài báo khoa học công bố tại các venue quốc tế về ứng dụng semantic communication và edge AI cho UAV.

# TÀI LIỆU THAM KHẢO

[1] M. Mueller, N. Smith, and B. Ghanem, "A Benchmark and Simulator for UAV Tracking," in Proc. ECCV, 2016.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. CVPR, 2016.

[3] Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, "Semantic Communications: Principles and Challenges," arXiv preprint, 2021.

[4] R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," Journal of Basic Engineering, 1960.

[5] Ryze Tech, "Tello SDK 2.0 User Guide," DJI/Ryze Tech Documentation.

*(Danh mục sẽ được bổ sung, cập nhật DOI/số trang đầy đủ trong quá trình triển khai đề tài.)*
