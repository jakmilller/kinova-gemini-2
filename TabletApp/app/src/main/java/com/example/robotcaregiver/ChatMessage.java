package com.example.robotcaregiver;

public class ChatMessage {

    private String message;
    private int isFromYou;

    public ChatMessage(String message, int isFromYou) {
        this.message = message;
        this.isFromYou = isFromYou;
    }

    public String getMessage() { return message; }
    public int isFromYou() { return isFromYou; }

}
