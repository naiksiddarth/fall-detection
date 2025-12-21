document.addEventListener('DOMContentLoaded', function() {
    const shutdown = document.getElementById("shutdown")
    if (shutdown) { // Added a check to ensure element exists
        shutdown.addEventListener('click', function () {
            fetch('/shutdown', { method: "POST" })
        })
    } else {
        console.error("Element with ID 'shutdown' not found.");
    }


    const toggleRecord = document.getElementById("record")
    if (toggleRecord) { // Added a check to ensure element exists
        toggleRecord.addEventListener('click', function () {
            fetch("/toggle_record", {
                method: "POST"
            })
                .then(response => response.json())
                .then(data => {
                    if (data.is_recording) {
                        toggleRecord.textContent = "Stop Recording"
                    } else {
                        toggleRecord.textContent = "Start Recording"
                    }
                })
        })
    } else {
        console.error("Element with ID 'record' not found.");
    }

    const reset = document.getElementById("reset")
    reset.addEventListener('click', function() {
        fetch('/reset', {
            method : "POST"
        })
    })
});
