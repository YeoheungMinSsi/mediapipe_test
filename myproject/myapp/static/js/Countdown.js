// 카운트다운 타이머 설정 및 시작
function startCountdown(duration) {
    let timer = duration, minutes, seconds;
    const countdownElement = document.getElementById('timer');
    const interval = setInterval(() => {
        minutes = parseInt(timer / 60, 10);
        seconds = parseInt(timer % 60, 10);

        minutes = minutes < 10 ? "0" + minutes : minutes;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        countdownElement.textContent = minutes + ":" + seconds;

        if (--timer < 0) {
            clearInterval(interval);
        }
    }, 1000);
}

// 버튼 클릭 시 타이머 시작 및 Python 코드 트리거
document.getElementById('startButton').addEventListener('click', () => {
    startCountdown(3600); // 1시간 카운트다운 시작
    document.getElementById('startButton').disabled = true; // 버튼 비활성화

    fetch('http://localhost:5000/start')  // Python 코드 트리거
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
});