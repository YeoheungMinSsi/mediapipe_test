// 웹캠 스트림 가져오기
async function getMediaStream() {
    try {   
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const videoElement = document.getElementById('videoElement');
        videoElement.srcObject = stream;
    } catch (error) {
        console.error('Error accessing media devices.', error);
    }
}

// 페이지 로드 후 웹캠 스트림 가져오기 함수 호출
window.onload = function() {
    getMediaStream();
};