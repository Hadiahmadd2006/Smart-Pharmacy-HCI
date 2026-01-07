import com.google.gson.*;
import TUIO.*;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class TUIOReceiver implements TuioListener {

    private static final String REACTIVISION_PATH =
            "C:\\Users\\ghare\\OneDrive\\Documents\\GitHub\\Smart-Pharmacy-HCI-Project\\reacTIVision-1.5.1-win64\\reacTIVision.exe";

    private Socket socket;
    private PrintWriter out;
    private TuioClient client;
    private HashMap<Integer, TuioObject> objects = new HashMap<>();
    private Map<Integer, Integer> markerMap = new HashMap<>();
    private Set<Integer> unmappedWarnings = new HashSet<>();

    public TUIOReceiver() {
        loadMarkerMap();
        connectToTuio();
        connectToPython();
    }

    private void loadMarkerMap() {
        Path mapPath = Paths.get("marker_map.json");
        if (!Files.exists(mapPath)) {
            try {
                Files.writeString(mapPath, "{}");
                System.out.println("[Map] Created marker_map.json (empty).");
            } catch (IOException e) {
                System.err.println("[Map] Failed to create marker_map.json: " + e.getMessage());
            }
            return;
        }
        try {
            String json = Files.readString(mapPath);
            Gson gson = new Gson();
            Map<String, Double> raw = gson.fromJson(json, Map.class);
            if (raw == null) {
                return;
            }
            for (Map.Entry<String, Double> entry : raw.entrySet()) {
                try {
                    int key = Integer.parseInt(entry.getKey());
                    int value = entry.getValue().intValue();
                    markerMap.put(key, value);
                } catch (NumberFormatException ignored) {
                    // Skip invalid keys.
                }
            }
            if (!markerMap.isEmpty()) {
                System.out.println("[Map] Loaded marker_map.json.");
            }
        } catch (IOException e) {
            System.err.println("[Map] Failed to read marker_map.json: " + e.getMessage());
        }
    }

    private int mapMarkerId(int rawId) {
        Integer mapped = markerMap.get(rawId);
        if (mapped != null) {
            return mapped;
        }
        if (unmappedWarnings.add(rawId)) {
            System.out.println("[Map] Unmapped marker " + rawId + " (using raw id).");
        }
        return rawId;
    }

    private void connectToTuio() {
        int attempts = 0;
        while (attempts < 30) {
            try {
                client = new TuioClient(3333);
                client.addTuioListener(this);
                client.connect();
                System.out.println("[TUIO] Receiver started, listening on port 3333...");
                return;
            } catch (Exception e) {
                attempts++;
                System.err.println("[TUIO] Failed to connect (retry " + attempts + "): " + e.getMessage());
                try {
                    Thread.sleep(500);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
        System.err.println("[TUIO] Giving up after " + attempts + " attempts.");
    }

    private void connectToPython() {
        int attempts = 0;
        while (attempts < 30) {
            try {
                socket = new Socket("127.0.0.1", 5055);
                out = new PrintWriter(socket.getOutputStream(), true);
                System.out.println("[Socket] Connected to Python server.");
                return;
            } catch (IOException e) {
                attempts++;
                System.err.println("[Socket] Connection failed (retry " + attempts + "): " + e.getMessage());
                try {
                    Thread.sleep(500);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
        System.err.println("[Socket] Giving up after " + attempts + " attempts.");
    }

    @Override
    public void addTuioObject(TuioObject obj) {
        int rawId = obj.getSymbolID();
        int mappedId = mapMarkerId(rawId);
        objects.put(rawId, obj);
        System.out.printf("[Add] Marker %d -> %d at (%.3f, %.3f) angle=%.2f%n",
                          rawId, mappedId, obj.getX(), obj.getY(), obj.getAngle());
        if (out != null) {
            JsonObject json = new JsonObject();
            json.addProperty("event", "add");
            json.addProperty("id", mappedId);
            json.addProperty("x", obj.getX());
            json.addProperty("y", obj.getY());
            json.addProperty("angle", obj.getAngle());
            out.println(json.toString());
            System.out.println("[Socket->Python] Sent add " + json.toString());
        }
    }

    @Override
    public void updateTuioObject(TuioObject obj) {
        int rawId = obj.getSymbolID();
        int mappedId = mapMarkerId(rawId);
        System.out.printf("[Update] Marker %d -> %d moved to (%.3f, %.3f) angle=%.2f%n",
                          rawId, mappedId, obj.getX(), obj.getY(), obj.getAngle());
        if (out != null) {
            JsonObject json = new JsonObject();
            json.addProperty("event", "update");
            json.addProperty("id", mappedId);
            json.addProperty("x", obj.getX());
            json.addProperty("y", obj.getY());
            json.addProperty("angle", obj.getAngle());
            out.println(json.toString());
            System.out.println("[Socket->Python] Sent update " + json.toString());
        }
    }

    @Override
    public void removeTuioObject(TuioObject obj) {
        int rawId = obj.getSymbolID();
        int mappedId = mapMarkerId(rawId);
        objects.remove(rawId);
        System.out.printf("[Remove] Marker %d -> %d removed%n", rawId, mappedId);
        if (out != null) {
            JsonObject json = new JsonObject();
            json.addProperty("event", "remove");
            json.addProperty("id", mappedId);
            out.println(json.toString());
            System.out.println("[Socket->Python] Sent remove " + json.toString());
        }
    }

    @Override
    public void addTuioCursor(TuioCursor tcur) {}
    @Override
    public void updateTuioCursor(TuioCursor tcur) {}
    @Override
    public void removeTuioCursor(TuioCursor tcur) {}
    @Override
    public void addTuioBlob(TuioBlob tblb) {}
    @Override
    public void updateTuioBlob(TuioBlob tblb) {}
    @Override
    public void removeTuioBlob(TuioBlob tblb) {}
    @Override
    public void refresh(TuioTime time) {}

    private static void startReactivision() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        if (!osName.contains("win")) {
            System.out.println("[TUIO] reacTIVision launch skipped (non-Windows).");
            return;
        }
        Path exePath = Paths.get(REACTIVISION_PATH);
        if (!Files.exists(exePath)) {
            System.out.println("[TUIO] reacTIVision not found: " + REACTIVISION_PATH);
            return;
        }
        try {
            ProcessBuilder builder = new ProcessBuilder(
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Start-Process -FilePath '" + REACTIVISION_PATH + "' -WindowStyle Minimized"
            );
            builder.start();
            System.out.println("[TUIO] reacTIVision started.");
        } catch (Exception exc) {
            System.out.println("[TUIO] Failed to start reacTIVision: " + exc.getMessage());
        }
    }

    public static void main(String[] args) {
        startReactivision();
        new TUIOReceiver();
        while (true) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                break;
            }
        }
    }
}
