package com.example.robotcaregiver;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class ChatAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private static final int VIEW_TYPE_YOU = 1;
    private static final int VIEW_TYPE_ROBOT = 2;
    private static final int VIEW_TYPE_PHONE = 3;

    private List<ChatMessage> messageList;

    public ChatAdapter(List<ChatMessage> messageList) {
        this.messageList = messageList;
    }

    @Override
    public int getItemViewType(int position) {
        ChatMessage message = messageList.get(position);
        return message.isFromYou();
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        if (viewType == VIEW_TYPE_YOU) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_you, parent, false);
            return new SentMessageViewHolder(view);
        } else if(viewType == VIEW_TYPE_ROBOT){
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_robot, parent, false);
            return new RobotMessageViewHolder(view);
        } else {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_phone, parent, false);
            return new PhoneMessageViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        ChatMessage message = messageList.get(position);
        if (holder.getItemViewType() == VIEW_TYPE_YOU) {
            ((SentMessageViewHolder) holder).bind(message);
        } else if (holder.getItemViewType() == VIEW_TYPE_ROBOT) {
            ((RobotMessageViewHolder) holder).bind(message);
        } else if (holder.getItemViewType() == VIEW_TYPE_PHONE) {
            ((PhoneMessageViewHolder) holder).bind(message);
        }
    }

    @Override
    public int getItemCount() {
        return messageList != null ? messageList.size() : 0;
    }

    static class SentMessageViewHolder extends RecyclerView.ViewHolder {
        TextView messageText;

        SentMessageViewHolder(View itemView) {
            super(itemView);
            messageText = itemView.findViewById(R.id.text_you);
        }

        void bind(ChatMessage message) {
            messageText.setText(message.getMessage());
        }
    }

    static class RobotMessageViewHolder extends RecyclerView.ViewHolder {
        TextView messageText;

        RobotMessageViewHolder(View itemView) {
            super(itemView);
            messageText = itemView.findViewById(R.id.text_robot);
        }

        void bind(ChatMessage message) {
            messageText.setText(message.getMessage());
        }
    }

    static class PhoneMessageViewHolder extends RecyclerView.ViewHolder {
        TextView messageText;

        PhoneMessageViewHolder(View itemView) {
            super(itemView);
            messageText = itemView.findViewById(R.id.text_phone);
        }

        void bind(ChatMessage message) {
            messageText.setText(message.getMessage());
        }
    }
}
