import re
import random
import datetime
import logging

logger = logging.getLogger(__name__)

class ConversationHandler:
    """
    Handles conversational queries that don't require knowledge from the database.
    Examples: greetings, small talk, system information, etc.
    """
    
    def __init__(self):
        # Define pattern groups for different types of conversational queries
        self.patterns = {
            # Greetings
            'greeting': [
                r'(?i)(xin\s+chào|chào\s+bạn|hello|hi|hey|xin\s+chào\s+bạn|chào|chao)',
                r'(?i)(buổi\s+sáng|buổi\s+chiều|buổi\s+tối)\s+(tốt\s+lành)',
                r'(?i)(good\s+morning|good\s+afternoon|good\s+evening)',
            ],
            
            # Farewells
            'farewell': [
                r'(?i)(tạm\s+biệt|goodbye|bye|see\s+you|gặp\s+lại\s+sau)',
                r'(?i)(hẹn\s+gặp\s+lại|hen\s+gap\s+lai)',
            ],
            
            # Health inquiries
            'health_inquiry': [
                r'(?i)(bạn\s+khỏe\s+không|khỏe\s+không|how\s+are\s+you|khoe\s+khong)',
                r'(?i)(sức\s+khỏe|dạo\s+này)',
                r'(?i)(bạn\s+có\s+khỏe|có\s+khỏe)',
            ],
            
            # Thank you
            'thanks': [
                r'(?i)(cảm\s+ơn|cám\s+ơn|thank|thanks|thank\s+you)',
                r'(?i)(cảm\s+ơn\s+nhiều|cảm\s+ơn\s+bạn|cám\s+ơn\s+bạn|cám\s+ơn\s+nhiều)',
            ],
            
            # Bot identity
            'bot_identity': [
                r'(?i)(bạn\s+là\s+ai|who\s+are\s+you|mày\s+là\s+ai|bạn\s+tên\s+gì|bạn\s+là\s+gì)',
                r'(?i)(bạn\s+làm\s+gì|công\s+việc|nhiệm\s+vụ)',
                r'(?i)(tên\s+bạn\s+là\s+gì|ban\s+ten\s+gi)',
            ],
            
            # System information
            'system_info': [
                r'(?i)(chatbot|hệ\s+thống|trợ\s+lý|assistant|bạn\s+hoạt\s+động|được\s+tạo)',
                r'(?i)(ai\s+tạo\s+ra\s+bạn|ai\s+là\s+người\s+tạo\s+ra\s+bạn)',
                r'(?i)(làm\s+thế\s+nào|how\s+do\s+you|bạn\s+hoạt\s+động\s+như\s+thế\s+nào)',
            ],
            
            # User emotions
            'user_emotion': [
                r'(?i)(mình\s+buồn|tôi\s+buồn|tôi\s+vui|mình\s+vui|tôi\s+lo\s+lắng)',
                r'(?i)(mệt\s+mỏi|mệt\s+quá|chán|cảm\s+thấy\s+chán|cảm\s+thấy|stress)',
            ],
            
            # Help request
            'help_request': [
                r'(?i)(giúp\s+đỡ|help|giúp\s+mình|giúp\s+tôi|cứu|hãy\s+giúp)',
                r'(?i)(bạn\s+có\s+thể\s+giúp|bạn\s+giúp\s+mình|ban\s+giup\s+minh)',
                r'(?i)(làm\s+thế\s+nào\s+để|làm\s+sao\s+để|how\s+to)',
            ],
            
            # Bot capabilities
            'bot_capabilities': [
                r'(?i)(bạn\s+có\s+thể\s+làm\s+gì|bạn\s+có\s+thể\s+giúp\s+những\s+gì|what\s+can\s+you\s+do)',
                r'(?i)(chức\s+năng|tính\s+năng|function|feature)',
                r'(?i)(bạn\s+biết\s+những\s+gì|bạn\s+biết\s+gì)',
            ],
            
            # University general questions
            'university_general': [
                r'(?i)(trường|đại\s+học|học\s+trường|university|truong)',
                r'(?i)(trường\s+đại\s+học\s+mở|trường\s+mở|dai\s+hoc\s+mo)',
            ],
            
            # Jokes and fun requests
            'jokes': [
                r'(?i)(kể\s+chuyện\s+cười|kể\s+joke|nói\s+chuyện\s+vui|tell\s+joke|tell\s+a\s+joke)',
                r'(?i)(chuyện\s+vui|truyện\s+cười|cười|vui|funny)',
            ],
            
            # Facts about university/education
            'fun_facts': [
                r'(?i)(fact|sự\s+thật\s+thú\s+vị|interesting\s+fact|điều\s+thú\s+vị)',
                r'(?i)(có\s+điều\s+gì\s+thú\s+vị|có\s+thông\s+tin\s+gì\s+thú\s+vị)',
            ],
            
            # Vietnamese-specific expressions
            'vietnamese_expressions': [
                r'(?i)(ơi\s+$|này$|ê\s+$|này\s+bạn|bạn\s+ơi)',
                r'(?i)(thế\s+nhỉ\?|đúng\s+không\?|phải\s+không\?|thế\s+à\?)',
                r'(?i)(bạn\s+hiểu\s+không|hiểu\s+chưa|hiểu\s+rồi|hiểu\s+không)',
            ],
            
            # Out of scope or off-topic questions
            'out_of_scope': [
                r'(?i)(thời\s+tiết|weather|dự\s+báo)',
                r'(?i)(bóng\s+đá|thể\s+thao|sport|lịch\s+thi\s+đấu)',
                r'(?i)(corona|covid|dịch\s+bệnh|vaccine)',
                r'(?i)(chính\s+trị|politics|chính\s+phủ|government)',
                r'(?i)(làm\s+quen|phone|tell\s+me\s+your\s+number|số\s+điện\s+thoại)',
            ],
        }
        
        # Define responses for each pattern type
        self.responses = {
            'greeting': [
                '<h4 class="text-gradient">Xin chào!</h4><p>Chào bạn! Mình là trợ lý tư vấn tuyển sinh của trường Đại học Mở Thành phố Hồ Chí Minh. Mình có thể giúp gì cho bạn?</p>',
                '<h4 class="text-gradient">Chào bạn!</h4><p>Rất vui được gặp bạn. Mình là trợ lý tư vấn tuyển sinh của trường Đại học Mở TP.HCM. Bạn cần tư vấn về vấn đề gì?</p>',
                '<h4 class="text-gradient">Xin chào!</h4><p>Chào mừng bạn đến với hệ thống tư vấn tuyển sinh trường Đại học Mở TP.HCM! Bạn cần tìm hiểu thông tin gì?</p>'
            ],
            
            'farewell': [
                '<h4 class="text-gradient">Tạm biệt!</h4><p>Cảm ơn bạn đã trò chuyện. Hẹn gặp lại bạn lần sau!</p>',
                '<h4 class="text-gradient">Tạm biệt!</h4><p>Rất vui được hỗ trợ bạn. Chúc bạn một ngày tốt lành!</p>',
                '<h4 class="text-gradient">Chào tạm biệt!</h4><p>Nếu còn thắc mắc gì, bạn có thể quay lại chat với mình bất cứ lúc nào nhé!</p>'
            ],
            
            'health_inquiry': [
                '<p>Cảm ơn bạn đã hỏi thăm! Mình là trợ lý ảo nên luôn trong trạng thái sẵn sàng phục vụ bạn. Bạn cần hỗ trợ thông tin gì về trường Đại học Mở TP.HCM?</p>',
                '<p>Mình là chatbot nên luôn khỏe mạnh và sẵn sàng hỗ trợ bạn 24/7. Bạn cần tìm hiểu thông tin gì về tuyển sinh?</p>',
                '<p>Mình hoạt động tốt, cảm ơn bạn đã quan tâm! Mình có thể giúp bạn tìm hiểu thông tin tuyển sinh của trường Đại học Mở không?</p>'
            ],
            
            'thanks': [
                '<p>Không có gì đâu bạn! Mình rất vui khi được giúp bạn. Bạn còn cần hỗ trợ gì nữa không?</p>',
                '<p>Rất vui khi thông tin mình cung cấp có ích cho bạn! Nếu còn thắc mắc gì, đừng ngại hỏi mình nhé.</p>',
                '<p>Không có chi! Đó là nhiệm vụ của mình. Bạn còn câu hỏi nào về tuyển sinh nữa không?</p>'
            ],
            
            'bot_identity': [
                '<h4 class="text-gradient">Về tôi</h4><p>Mình là Admission ChatGenie - trợ lý tư vấn tuyển sinh ảo của trường Đại học Mở Thành phố Hồ Chí Minh. Mình được tạo ra để hỗ trợ thông tin tuyển sinh và giải đáp thắc mắc cho các bạn thí sinh.</p>',
                '<h4 class="text-gradient">Trợ lý tư vấn tuyển sinh</h4><p>Mình là chatbot tư vấn tuyển sinh của trường Đại học Mở TP.HCM, được tạo ra để giúp các bạn tìm hiểu thông tin về ngành học, điểm chuẩn, học phí và các thông tin tuyển sinh khác.</p>',
                '<h4 class="text-gradient">Giới thiệu</h4><p>Xin chào! Mình là trợ lý ảo chuyên cung cấp thông tin tuyển sinh của trường Đại học Mở Thành phố Hồ Chí Minh. Bạn có thể hỏi mình bất kỳ thông tin gì liên quan đến tuyển sinh của nhà trường.</p>'
            ],
            
            'system_info': [
                '<h4 class="text-gradient">Về hệ thống</h4><p>Mình là một chatbot thông minh được phát triển dựa trên công nghệ Generative AI kết hợp với Retrieval-Augmented Generation (RAG). Mình hoạt động bằng cách tìm kiếm thông tin liên quan từ cơ sở dữ liệu tài liệu PDF của trường và sử dụng AI để tổng hợp câu trả lời chính xác.</p>',
                '<h4 class="text-gradient">Cách mình hoạt động</h4><p>Mình là chatbot sử dụng công nghệ xử lý ngôn ngữ tự nhiên và tìm kiếm thông tin theo ngữ cảnh. Khi bạn đặt câu hỏi, mình sẽ tìm kiếm thông tin từ cơ sở dữ liệu tài liệu của trường Đại học Mở và đưa ra câu trả lời phù hợp nhất.</p>',
                '<h4 class="text-gradient">Công nghệ sử dụng</h4><p>Mình được xây dựng dựa trên các công nghệ tiên tiến như Vector Database, LLM (Large Language Model), và kỹ thuật xử lý ngôn ngữ tự nhiên. Mình liên tục học hỏi để cải thiện khả năng trả lời của mình.</p>'
            ],
            
            'user_emotion': [
                '<p>Mình hiểu cảm xúc của bạn. Quá trình tìm hiểu thông tin tuyển sinh đôi khi khá căng thẳng. Mình ở đây để giúp bạn tìm hiểu thông tin một cách dễ dàng hơn. Bạn cần hỗ trợ gì?</p>',
                '<p>Cảm xúc của bạn rất quan trọng. Nếu bạn cảm thấy lo lắng về kỳ thi hoặc việc chọn trường, mình có thể giúp cung cấp thông tin chính xác để bạn yên tâm hơn. Bạn muốn biết gì về trường Đại học Mở?</p>',
                '<p>Mình ở đây để giúp bạn. Hãy cho mình biết bạn cần tìm hiểu thông tin gì, mình sẽ cố gắng giúp bạn giải đáp mọi thắc mắc về tuyển sinh.</p>'
            ],
            
            'help_request': [
                '<h4 class="text-gradient">Mình có thể giúp gì?</h4><p>Mình có thể giúp bạn tìm hiểu về các thông tin tuyển sinh như: điểm chuẩn các năm, chỉ tiêu tuyển sinh, học phí, chương trình đào tạo, các ngành học, học bổng, cơ sở vật chất, và nhiều thông tin khác. Bạn muốn biết về vấn đề nào?</p>',
                '<h4 class="text-gradient">Sẵn sàng hỗ trợ!</h4><p>Mình có thể giúp bạn giải đáp các thắc mắc về tuyển sinh của trường Đại học Mở TP.HCM. Bạn hãy đặt câu hỏi cụ thể về điều bạn muốn biết nhé!</p>',
                '<h4 class="text-gradient">Tôi có thể giúp gì?</h4><p>Mình sẵn sàng hỗ trợ bạn tìm hiểu thông tin tuyển sinh. Bạn đang quan tâm đến ngành nào hoặc có thắc mắc gì về quá trình tuyển sinh?</p>'
            ],
            
            'bot_capabilities': [
                '<h4 class="text-gradient">Khả năng của tôi</h4><p>Mình có thể giúp bạn:</p><ul><li>Tra cứu điểm chuẩn các ngành</li><li>Cung cấp thông tin về học phí</li><li>Tư vấn về các ngành đào tạo</li><li>Giải đáp thắc mắc về chương trình học</li><li>Thông tin về học bổng</li><li>Hướng dẫn thủ tục xét tuyển</li><li>Thông tin về cơ sở vật chất</li></ul>',
                '<h4 class="text-gradient">Tôi có thể làm gì?</h4><p>Mình là trợ lý tư vấn tuyển sinh, có thể giúp bạn:</p><ul><li>Tìm hiểu về các phương thức xét tuyển</li><li>Cung cấp thông tin chi tiết về các ngành học</li><li>Tra cứu điểm chuẩn qua các năm</li><li>Thông tin về học phí và học bổng</li><li>Giải đáp các thắc mắc về quy trình nhập học</li><li>Cung cấp thông tin về cơ sở vật chất của trường</li></ul>',
                '<h4 class="text-gradient">Chức năng của tôi</h4><p>Mình được tạo ra để giúp bạn tìm hiểu mọi thông tin liên quan đến tuyển sinh của trường Đại học Mở TP.HCM như điểm chuẩn, ngành học, học phí, học bổng và các thông tin khác. Bạn cứ hỏi, mình sẽ cố gắng trả lời!</p>'
            ],
            
            'university_general': [
                '<h4 class="text-gradient">Trường Đại học Mở TP.HCM</h4><p>Trường Đại học Mở Thành phố Hồ Chí Minh là một trường đại học công lập trực thuộc Bộ Giáo dục và Đào tạo. Trường có nhiều ngành học đa dạng, với các phương thức đào tạo linh hoạt, môi trường học tập năng động. Bạn muốn biết thông tin cụ thể nào về trường?</p>',
                '<h4 class="text-gradient">Giới thiệu về Trường</h4><p>Trường Đại học Mở TP.HCM được thành lập từ năm 1990 và là một trong những trường đại học hàng đầu tại TP.HCM. Trường có nhiều ngành đào tạo chất lượng và cơ sở vật chất hiện đại. Bạn cần tư vấn cụ thể về vấn đề nào?</p>',
                '<h4 class="text-gradient">ĐH Mở TP.HCM</h4><p>Trường Đại học Mở Thành phố Hồ Chí Minh có nhiều chương trình đào tạo chất lượng, đội ngũ giảng viên giỏi và cơ sở vật chất hiện đại. Trường cung cấp môi trường học tập năng động và cơ hội việc làm rộng mở sau khi tốt nghiệp. Bạn muốn tìm hiểu thêm về điểm chuẩn, ngành học hay học phí?</p>'
            ],
            
            'jokes': [
                '<h4 class="text-gradient">Chuyện vui về trường đại học</h4><p>Một sinh viên gọi về nhà: "Mẹ ơi, con đã tiêu hết tiền học kỳ này rồi!" Mẹ trả lời: "Con đã làm gì với khoản tiền đó vậy? Học kỳ mới mới bắt đầu được 2 tuần!" Sinh viên: "Con biết... Nhưng trường đại học in danh sách sinh viên bị đuổi học sớm quá, và con phải trả phí để không có tên trong đó!" 😂</p>',
                '<h4 class="text-gradient">Chuyện cười</h4><p>Giáo sư hỏi sinh viên: "Tại sao bạn lại nộp một tờ giấy trắng làm bài kiểm tra?" Sinh viên trả lời: "Thưa thầy, đó là vì em và thầy đều biết câu trả lời, nên em không muốn lặp lại những điều hiển nhiên ạ!" 😁</p>',
                '<h4 class="text-gradient">Nụ cười tuyển sinh</h4><p>Thí sinh: "Thầy ơi, em muốn học ngành không phải làm nhiều bài tập về nhà, không cần thi cử nhiều, và ra trường có việc làm lương cao ngay. Trường mình có ngành nào như thế không ạ?" Thầy tư vấn tuyển sinh: "Có chứ, đó gọi là... giấc mơ!" 🤣</p>'
            ],
            
            'fun_facts': [
                '<h4 class="text-gradient">Sự thật thú vị về giáo dục</h4><p>Bạn có biết: Trường đại học đầu tiên trên thế giới là Đại học Al-Qarawiyyin ở Morocco, được thành lập vào năm 859! Còn trường đại học đầu tiên ở Việt Nam là Quốc Tử Giám - tiền thân của Đại học Quốc gia Hà Nội ngày nay, được thành lập từ thời nhà Lý (1076).</p>',
                '<h4 class="text-gradient">Điều thú vị về Đại học Mở</h4><p>Trường Đại học Mở TP.HCM là một trong những trường đại học đi đầu trong đào tạo từ xa tại Việt Nam. Trường còn có hệ thống học trực tuyến hiện đại giúp sinh viên có thể học mọi lúc, mọi nơi. Đặc biệt, nhiều chương trình đào tạo của trường có sự hợp tác với các trường đại học uy tín trên thế giới.</p>',
                '<h4 class="text-gradient">Fact thú vị</h4><p>Sinh viên học đại học thường thay đổi ngành học ít nhất 3 lần trước khi tốt nghiệp! Đó là lý do tại sao việc tìm hiểu kỹ về ngành học trước khi đăng ký rất quan trọng. Trường Đại học Mở TP.HCM cung cấp nhiều buổi tư vấn hướng nghiệp giúp các bạn định hướng ngành học phù hợp.</p>'
            ],
            
            'vietnamese_expressions': [
                '<p>Dạ! Mình đây, bạn cần giúp gì về thông tin tuyển sinh của trường Đại học Mở TP.HCM?</p>',
                '<p>Mình đang nghe nè! Bạn muốn tìm hiểu thông tin gì về trường Đại học Mở TP.HCM?</p>',
                '<p>Dạ, mình hiểu rồi ạ! Bạn cần mình giải đáp thông tin gì về tuyển sinh?</p>'
            ],
            
            'out_of_scope': [
                '<h4 class="text-gradient">Xin lỗi bạn!</h4><p>Mình là trợ lý tư vấn tuyển sinh của trường Đại học Mở TP.HCM, nên mình chỉ có thể trả lời các câu hỏi liên quan đến tuyển sinh, ngành học, điểm chuẩn, học phí và các thông tin về trường thôi. Bạn có thể hỏi mình những thông tin này nhé!</p>',
                '<h4 class="text-gradient">Thông tin ngoài phạm vi</h4><p>Rất tiếc, câu hỏi của bạn nằm ngoài phạm vi kiến thức của mình. Mình chỉ có thể tư vấn về các vấn đề liên quan đến tuyển sinh của trường Đại học Mở TP.HCM như: điểm chuẩn, ngành học, học phí, quy trình xét tuyển, v.v. Bạn có thể hỏi mình về những chủ đề này không?</p>',
                '<h4 class="text-gradient">Chủ đề khác</h4><p>Mình là trợ lý chuyên về tuyển sinh của trường Đại học Mở TP.HCM nên không thể trả lời câu hỏi này. Bạn có thể hỏi mình về các chủ đề như: chỉ tiêu tuyển sinh, điểm chuẩn, học phí, chương trình đào tạo, hoặc cơ sở vật chất của trường Đại học Mở TP.HCM.</p>'
            ],
            
        }
    
    def get_current_time_greeting(self):
        """Returns a greeting based on the current time of day"""
        current_hour = datetime.datetime.now().hour
        
        if 5 <= current_hour < 12:
            time_greeting = "Chào buổi sáng"
        elif 12 <= current_hour < 18:
            time_greeting = "Chào buổi chiều"
        else:
            time_greeting = "Chào buổi tối"
            
        return f'<p>{time_greeting}! Mình là trợ lý tư vấn tuyển sinh của trường Đại học Mở TP.HCM. Mình có thể giúp gì cho bạn?</p>'
    
    def detect_query_type(self, query):
        """
        Detects the type of conversational query based on predefined patterns.
        Returns the query type if found, None otherwise.
        """
        if not query:
            return None

        # Trước tiên, kiểm tra các pattern tìm kiếm thông tin
        # Nếu câu hỏi chứa các từ khóa yêu cầu thông tin, ưu tiên phân loại là không phải hội thoại đơn thuần
        information_seeking_patterns = [
            r'(?i)(thông\s+tin|tư\s+vấn|cho\s+biết|cho\s+hỏi)',
            r'(?i)(điểm\s+chuẩn|học\s+phí|tuyển\s+sinh|xét\s+tuyển|kỳ\s+thi|học\s+bổng)',
            r'(?i)(hiệu\s+trưởng|phó\s+hiệu\s+trưởng|trưởng\s+khoa|giảng\s+viên|thí\s+sinh|sinh\s+viên)',
            r'(?i)(ngành\s|chuyên\s+ngành|ngành\s+học|khoa\s|tốt\s+nghiệp|kiến\s+thức|đào\s+tạo|chương\s+trình|trang\s+bị)',
            r'(?i)(hồ\s+sơ|phương\s+thức|giấy\s+tờ|thủ\s+tục|đăng\s+ký)',
            r'(?i)(mấy\s+điểm|bao\s+nhiêu\s+điểm|số\s+điểm|mức\s+điểm|lệ\s+phí|điểm\s+đầu\s+vào)',
            r'(?i)(mấy\s+tiền|bao\s+nhiêu\s+tiền|chi\s+phí|tốn|đóng)',
            r'(?i)(khi\s+nào|lúc\s+nào|thời\s+hạn|hạn\s+chót|deadline)',
            r'(?i)(được\s+không|có\s+được|có\s+thể|có\s+cần|liệu\s+có|gì\s)',
            r'(?i)(việc\s+làm|cơ\s+hội|tương\s+lai|ra\s+trường|sau\s+khi\s+học)',
        ]
        
        for pattern in information_seeking_patterns:
            if re.search(pattern, query):
                logger.debug(f"Query contains information-seeking pattern: {pattern}")
                # Đây là câu hỏi tìm kiếm thông tin, không xem là hội thoại đơn thuần
                return None

        standard_query_types = [qt for qt in self.patterns.keys() if qt != 'out_of_scope']
        
        logger.debug(f"Checking query: '{query}'")
        
        # Then check standard conversation patterns
        for query_type in standard_query_types:
            for pattern in self.patterns[query_type]:
                if re.search(pattern, query):
                    logger.debug(f"Matched standard query type: {query_type} with pattern: {pattern}")
                    return query_type
        
        # Finally check out of scope patterns
        if 'out_of_scope' in self.patterns:
            for pattern in self.patterns['out_of_scope']:
                if re.search(pattern, query):
                    logger.debug(f"Matched out of scope query with pattern: {pattern}")
                    return 'out_of_scope'
        
        logger.debug("No pattern match found for query")
        return None
    
    def is_conversational_query(self, query):
        """Returns True if the query is conversational, False otherwise."""
        # First check specific patterns
        if self.detect_query_type(query) is not None:
            return True
            
        # Then check if it's likely an out-of-scope query
        if self.is_likely_out_of_scope(query):
            return True
            
        return False
    
    def is_likely_out_of_scope(self, query):
        """
        Checks if a query is likely out of scope by looking for the absence of 
        education-related keywords.
        """
        if not query or len(query) < 5:
            return False
            
        # List of education and university admissions related keywords
        education_keywords = [
            # Vietnamese keywords
            'đại học', 'trường', 'cao đẳng', 'tuyển sinh', 'ngành', 'điểm', 'học phí', 
            'xét tuyển', 'chỉ tiêu', 'học bổng', 'sinh viên', 'đào tạo',
            'cơ sở', 'vật chất', 'tín chỉ', 'khoa', 'chuyên ngành', 'trúng tuyển',
            'nhập học', 'cử nhân', 'tốt nghiệp', 'giảng viên', 'học kỳ', 'lớp',
            'môn học', 'bằng cấp', 'thạc sĩ', 'tiến sĩ', 'nghiên cứu', 'kỳ thi',
            'công nhận', 'mở', 'hồ chí minh',
            
            # English keywords
            'university', 'college', 'admission', 'major', 'score', 'tuition',
            'scholarship', 'student', 'education', 'faculty', 'graduate',
            'bachelor', 'master', 'phd', 'academic', 'semester', 'course',          
        ]
        
        # Check if query contains any education-related keywords
        query_lower = query.lower()
        for keyword in education_keywords:
            if keyword in query_lower:
                return False
                
        # Check if query length is too long (likely a specific question)
        if len(query.split()) >= 10:
            # Long queries without education keywords are likely out of scope
            return True
            
        # For shorter queries, be more conservative - don't assume out of scope
        return False
        
    def get_response(self, query):
        """
        Generates a response based on the query type.
        Returns a response if the query is conversational, None otherwise.
        """
        query_type = self.detect_query_type(query)
        logger.debug(f"Query: '{query}', Detected type: {query_type}")
        
        # If it's a known query type
        if query_type:
            # If it's a greeting, include time-based greeting occasionally
            if query_type == 'greeting' and random.random() < 0.3:
                return self.get_current_time_greeting()
                
            # Get a random response for the query type
            responses = self.responses.get(query_type, [])
            if responses:   
                selected_response = random.choice(responses)
                logger.debug(f"Selected random response from type {query_type}")
                return selected_response
        
        # If it's likely an out of scope query but doesn't match any specific pattern
        if self.is_likely_out_of_scope(query):
            logger.debug("Query identified as likely out of scope")
            return random.choice(self.responses.get('out_of_scope', []))
        
        logger.debug("No matching response found")    
        return None