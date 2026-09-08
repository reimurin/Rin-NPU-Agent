// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.PowerManager
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.AdapterView
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.ProgressBar
import android.widget.SimpleAdapter
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.GravityCompat
import androidx.core.widget.doAfterTextChanged
import androidx.fragment.app.FragmentActivity
import androidx.recyclerview.widget.RecyclerView
import com.geniex.demo.agent.AgentToolExecutor
import com.geniex.demo.agent.AgentToolParser
import com.geniex.demo.agent.ProjectWorkspace
import com.geniex.demo.agent.SharedAttachment
import com.geniex.demo.agent.SharedInputManager
import com.geniex.demo.bean.ModelData
import com.geniex.demo.bean.getSupportPluginIds
import com.geniex.demo.bean.isNpuModel
import com.geniex.demo.databinding.ActivityMainBinding
import com.geniex.demo.databinding.DialogSelectPluginIdBinding
import com.geniex.demo.listeners.CustomDialogInterface
import com.geniex.demo.history.ProjectSessionStore
import com.geniex.demo.history.StoredConversation
import com.geniex.demo.history.StoredProject
import com.geniex.demo.image.ImageModeController
import com.geniex.demo.image.ModelPackUpdateManager
import com.geniex.demo.image.ModelUpdateActivity
import com.geniex.demo.image.ModelUpdateCheck
import com.geniex.demo.image.RinControls
import com.geniex.demo.model.ModelLibraryActivity
import com.geniex.demo.model.RecommendedModels
import com.geniex.demo.model.TestModelSeeder
import com.geniex.demo.utils.ExecShell
import com.geniex.demo.utils.GgufVisionConfig
import com.geniex.demo.utils.GgufVisionReader
import com.geniex.demo.utils.ImgUtil
import com.geniex.demo.utils.UiErrorLocalizer
import com.geniex.demo.utils.inflate
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.ComputeUnitValue
import com.geniex.sdk.bean.HubSource
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.ModelPullInput
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmContent
import com.geniex.sdk.bean.VlmCreateInput
import com.gyf.immersionbar.ktx.immersionBar
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.abs

class MainActivity : FragmentActivity() {
    private val binding: ActivityMainBinding by inflate()
    private var downloadJob: Job? = null
    private var downloadingModelData: ModelData? = null
    private lateinit var llDownloading: LinearLayout
    private lateinit var tvDownloadProgress: TextView
    private lateinit var pbDownloading: ProgressBar
    private lateinit var spModelList: Spinner
    private lateinit var btnDownload: Button
    private lateinit var btnLoadModel: Button
    private lateinit var btnUnloadModel: Button
    private lateinit var btnStop: Button
    private lateinit var etInput: EditText
    private lateinit var btnSend: ImageButton
    private lateinit var btnClearHistory: Button
    private lateinit var btnAddImage: ImageButton

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ChatAdapter

    private lateinit var scrollImages: HorizontalScrollView
    private lateinit var topScrollContainer: LinearLayout
    private lateinit var llLoading: LinearLayout
    private lateinit var vTip: View

    private lateinit var llmWrapper: LlmWrapper
    private lateinit var vlmWrapper: VlmWrapper
    private val modelScope = CoroutineScope(Dispatchers.IO)

    private val chatList = arrayListOf<ChatMessage>()
    private val vlmChatList = arrayListOf<VlmChatMessage>()
    private lateinit var modelList: List<ModelData>
    private var catalogModels: List<ModelData> = emptyList()
    private var selectModelId = ""
    private var followLatestMessages = true
    private var chatUserDragging = false
    private var drawerSwipeDownX = 0f
    private var drawerSwipeDownY = 0f
    private var drawerSwipeTriggered = false

    private var isLoadLlmModel = false
    private var isLoadVlmModel = false

    /**
     * Vision geometry of the currently loaded VLM, read from its mmproj GGUF.
     * Null for LLM-only models, and when the mmproj declares nothing usable —
     * image preprocessing then falls back to [FALLBACK_VLM_IMAGE_SIZE].
     */
    private var vlmVisionConfig: GgufVisionConfig? = null

    private var enableThinking = false
    private var isGenerating = false

    private val savedImageFiles = mutableListOf<File>()
    private val messages = arrayListOf<Message>()
    private var loadingMessageIndex: Int = -1
    private lateinit var projectWorkspace: ProjectWorkspace
    private lateinit var sharedInputManager: SharedInputManager
    private lateinit var agentTools: AgentToolExecutor
    private val pendingSharedAttachments = mutableListOf<SharedAttachment>()
    private var agentModeEnabled = false
    private var suppressAgentSwitchCallback = false
    private var pendingEnableAgentAfterPicker = false
    private val agentUiPrefs by lazy { getSharedPreferences("rin_agent_ui", MODE_PRIVATE) }
    private lateinit var sessionStore: ProjectSessionStore
    private var currentProjectId: String = ""
    private var currentConversationId: String = ""
    private var crossConversationReadAuthorized = false
    private lateinit var imageModeController: ImageModeController
    private var startupModelUpdateChecked = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        immersionBar {
            statusBarColorInt(Color.WHITE)
            statusBarDarkFont(true)
        }
        sessionStore = ProjectSessionStore(this)
        projectWorkspace = ProjectWorkspace(this, sessionStore)
        sharedInputManager = SharedInputManager(this)
        agentTools = AgentToolExecutor(projectWorkspace, sessionStore) { crossConversationReadAuthorized }
        initData()
        initView()
        setListeners()
        setupProjectConversationUi()
        setupAgentUi()
        imageModeController = ImageModeController(this, binding, ::releaseChatRuntimeForImage)
        imageModeController.setup()
        checkModelPackUpdateOnLaunch()
        handleIncomingShare(intent)
    }

    private fun checkModelPackUpdateOnLaunch() {
        if (startupModelUpdateChecked) return
        startupModelUpdateChecked = true
        val updater = ModelPackUpdateManager(this)
        updater.checkAsync { result ->
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                when (result) {
                    is ModelUpdateCheck.UpdateAvailable -> {
                        val local = result.local?.version ?: getString(R.string.model_update_not_installed)
                        RinControls.dialog(this)
                            .setTitle(R.string.model_update_prompt_title)
                            .setMessage(getString(R.string.model_update_prompt_message, local, result.remote.version, formatModelUpdateBytes(result.remote.totalDownloadBytes)))
                            .setNegativeButton(R.string.model_update_later, null)
                            .setNeutralButton(R.string.model_update_ignore_version) { _, _ -> updater.ignoreVersion(result.remote) }
                            .setPositiveButton(R.string.model_update_now) { _, _ ->
                                startActivity(Intent(this, ModelUpdateActivity::class.java).putExtra(ModelUpdateActivity.EXTRA_AUTO_INSTALL, true))
                            }
                            .show()
                    }
                    is ModelUpdateCheck.AppUpdateRequired -> {
                        RinControls.dialog(this)
                            .setTitle(R.string.model_update_app_prompt_title)
                            .setMessage(getString(R.string.model_update_app_prompt_message, result.remote.version, formatModelUpdateBytes(result.remote.totalDownloadBytes)))
                            .setNegativeButton(android.R.string.cancel, null)
                            .setNeutralButton(R.string.model_update_ignore_version) { _, _ -> updater.ignoreVersion(result.remote) }
                            .setPositiveButton(R.string.model_update_open) { _, _ -> startActivity(Intent(this, ModelUpdateActivity::class.java)) }
                            .show()
                    }
                    else -> Unit
                }
            }
        }
    }

    private fun formatModelUpdateBytes(bytes: Long): String {
        val value = bytes.coerceAtLeast(0L).toDouble()
        return when {
            value >= 1073741824.0 -> String.format(Locale.US, "%.2f GiB", value / 1073741824.0)
            value >= 1048576.0 -> String.format(Locale.US, "%.1f MiB", value / 1048576.0)
            value >= 1024.0 -> String.format(Locale.US, "%.1f KiB", value / 1024.0)
            else -> "${bytes.coerceAtLeast(0L)} B"
        }
    }

    private fun setupProjectConversationUi() {
        binding.btnNewProject.setOnClickListener { showCreateProjectDialog() }
        binding.btnNewConversation.setOnClickListener { startNewConversation() }
        val project = sessionStore.loadCurrentProjectOrCreate()
        currentProjectId = project.id
        restoreConversation(sessionStore.loadCurrentConversationOrCreate(project.id), resetNative = false)
        refreshProjectList()
        refreshConversationList()
        refreshWorkspaceUi()
    }

    private fun showCreateProjectDialog() {
        val input = EditText(this).apply {
            hint = getString(R.string.project_name_hint)
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.new_project)
            .setView(input)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                persistCurrentConversation()
                val project = sessionStore.createProject(input.text.toString())
                switchProject(project)
                Toast.makeText(this, getString(R.string.project_created), Toast.LENGTH_SHORT).show()
            }.show()
    }

    private fun switchProject(project: StoredProject) {
        persistCurrentConversation()
        sessionStore.selectProject(project.id) ?: return
        currentProjectId = project.id
        val conversation = sessionStore.loadCurrentConversationOrCreate(project.id)
        restoreConversation(conversation, resetNative = true)
        val status = projectWorkspace.inspect()
        setAgentEnabled(agentUiPrefs.getBoolean(KEY_AGENT_ENABLED, false) && status.ready, persist = false)
        refreshWorkspaceUi(status)
        refreshProjectList()
        refreshConversationList()
    }

    private fun refreshProjectList() {
        if (!::sessionStore.isInitialized) return
        val projects = sessionStore.listProjects(50)
        val current = sessionStore.loadProject(currentProjectId)
        binding.tvCurrentProject.text = current?.let { getString(R.string.current_project, it.name) } ?: getString(R.string.current_project_none)
        binding.llProjectList.removeAllViews()
        projects.forEach { summary ->
            val row = TextView(this).apply {
                text = "${summary.name}\n${getString(R.string.project_conversation_count, summary.conversationCount)}"
                textSize = 14f
                maxLines = 3
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.rin_text_primary))
                setPadding(dp(12), dp(9), dp(12), dp(9))
                setTypeface(null, if (summary.id == currentProjectId) Typeface.BOLD else Typeface.NORMAL)
                setBackgroundResource(if (summary.id == currentProjectId) R.drawable.rin_bg_mode_selected else R.drawable.rin_bg_mode_unselected)
                setOnClickListener {
                    sessionStore.selectProject(summary.id)?.let(::switchProject)
                }
                setOnLongClickListener {
                    showProjectActions(summary.id, summary.name)
                    true
                }
            }
            binding.llProjectList.addView(row, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { bottomMargin = dp(4) })
        }
    }

    private fun showProjectActions(id: String, name: String) {
        AlertDialog.Builder(this)
            .setItems(arrayOf(getString(R.string.rename_project), getString(R.string.delete_project))) { _, which ->
                if (which == 0) showRenameProject(id, name) else confirmDeleteProject(id)
            }.show()
    }

    private fun showRenameProject(id: String, name: String) {
        val input = EditText(this).apply {
            setText(name)
            setSelection(text.length)
            hint = getString(R.string.project_name_hint)
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.rename_project)
            .setView(input)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.rename) { _, _ ->
                sessionStore.renameProject(id, input.text.toString())
                refreshProjectList()
            }.show()
    }

    private fun confirmDeleteProject(id: String) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_project)
            .setMessage(R.string.delete_project_confirm)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.delete) { _, _ ->
                persistCurrentConversation()
                sessionStore.loadProject(id)?.let { project ->
                    project.workspaceUri?.let { uriString ->
                        if (project.workspaceFlags != 0) runCatching {
                            contentResolver.releasePersistableUriPermission(Uri.parse(uriString), project.workspaceFlags)
                        }
                    }
                }
                val wasCurrent = id == currentProjectId
                val current = sessionStore.deleteProject(id)
                if (wasCurrent) switchProject(current) else {
                    refreshProjectList()
                    refreshConversationList()
                }
                Toast.makeText(this, getString(R.string.project_deleted), Toast.LENGTH_SHORT).show()
            }.show()
    }

    private fun startNewConversation() {
        persistCurrentConversation()
        val conversation = sessionStore.createConversation(currentProjectId)
        restoreConversation(conversation, resetNative = true)
        binding.drawerLayout.closeDrawer(GravityCompat.START)
    }

    private fun restoreConversation(conversation: StoredConversation, resetNative: Boolean) {
        currentConversationId = conversation.id
        loadingMessageIndex = -1
        messages.clear()
        messages.addAll(sessionStore.toMessages(conversation))
        clearImages()
        rebuildTemplateHistories()
        adapter.notifyDataSetChanged()
        followLatestMessages = true
        scrollChatToBottomIfFollowing(force = true)
        if (resetNative) resetNativeConversationState()
        refreshProjectList()
        refreshConversationList()
    }

    private fun rebuildTemplateHistories() {
        chatList.clear()
        vlmChatList.clear()
        messages.forEach { message ->
            when (message.type) {
                MessageType.USER -> {
                    chatList.add(ChatMessage("user", message.content))
                    vlmChatList.add(VlmChatMessage("user", listOf(VlmContent("text", message.content))))
                }
                MessageType.ASSISTANT -> {
                    chatList.add(ChatMessage("assistant", message.content))
                    vlmChatList.add(VlmChatMessage("assistant", listOf(VlmContent("text", message.content))))
                }
                else -> Unit
            }
        }
    }

    private fun resetNativeConversationState() {
        modelScope.launch {
            if (isLoadLlmModel) runCatching { llmWrapper.reset() }
            if (isLoadVlmModel) runCatching { vlmWrapper.reset() }
        }
    }

    private fun persistCurrentConversation() {
        if (!::sessionStore.isInitialized || currentProjectId.isBlank() || currentConversationId.isBlank()) return
        runCatching { sessionStore.saveConversation(currentProjectId, currentConversationId, messages.toList()) }
            .onFailure { Log.e(TAG, "conversation save failed", it) }
        refreshProjectList()
        refreshConversationList()
    }

    private fun refreshConversationList() {
        if (!::sessionStore.isInitialized || currentProjectId.isBlank()) return
        val summaries = sessionStore.listConversations(currentProjectId, 50)
        binding.llConversationList.removeAllViews()
        binding.tvHistoryEmpty.visibility = if (summaries.isEmpty()) View.VISIBLE else View.GONE
        summaries.forEach { summary ->
            val row = TextView(this).apply {
                text = summary.title
                textSize = 14f
                maxLines = 2
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.rin_text_primary))
                setPadding(dp(12), dp(10), dp(12), dp(10))
                setTypeface(null, if (summary.id == currentConversationId) Typeface.BOLD else Typeface.NORMAL)
                setBackgroundResource(if (summary.id == currentConversationId) R.drawable.rin_bg_mode_selected else R.drawable.rin_bg_mode_unselected)
                setOnClickListener {
                    persistCurrentConversation()
                    sessionStore.selectConversation(currentProjectId, summary.id)?.let { restoreConversation(it, resetNative = true) }
                    binding.drawerLayout.closeDrawer(GravityCompat.START)
                }
                setOnLongClickListener {
                    showConversationActions(summary.id, summary.title)
                    true
                }
            }
            binding.llConversationList.addView(row, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { bottomMargin = dp(4) })
        }
    }

    private fun showConversationActions(id: String, title: String) {
        AlertDialog.Builder(this)
            .setItems(arrayOf(getString(R.string.rename_conversation), getString(R.string.delete_conversation))) { _, which ->
                if (which == 0) showRenameConversation(id, title) else confirmDeleteConversation(id)
            }.show()
    }

    private fun showRenameConversation(id: String, title: String) {
        val input = EditText(this).apply {
            setText(title)
            setSelection(text.length)
            hint = getString(R.string.conversation_title_hint)
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.rename_conversation)
            .setView(input)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.rename) { _, _ ->
                sessionStore.renameConversation(currentProjectId, id, input.text.toString())
                refreshConversationList()
                refreshProjectList()
            }.show()
    }

    private fun confirmDeleteConversation(id: String) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_conversation)
            .setMessage(R.string.delete_conversation_confirm)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.delete) { _, _ ->
                val wasCurrent = id == currentConversationId
                val replacement = sessionStore.deleteConversation(currentProjectId, id)
                if (wasCurrent) restoreConversation(replacement, resetNative = true) else {
                    refreshConversationList()
                    refreshProjectList()
                }
            }.show()
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun setupAgentUi() {
        binding.btnChooseWorkspace.setOnClickListener { launchWorkspacePicker(false) }
        binding.btnTestWorkspace.setOnClickListener {
            modelScope.launch {
                val status = projectWorkspace.probeAccess()
                runOnUiThread {
                    refreshWorkspaceUi(status)
                    Toast.makeText(this@MainActivity, if (status.ready) getString(R.string.workspace_access_ok) else getString(R.string.workspace_access_failed, status.detail ?: getString(R.string.workspace_permission_failed)), Toast.LENGTH_LONG).show()
                }
            }
        }
        binding.btnClearWorkspace.setOnClickListener {
            setAgentEnabled(false)
            projectWorkspace.clearRoot()
            refreshWorkspaceUi()
            Toast.makeText(this, getString(R.string.workspace_cleared), Toast.LENGTH_SHORT).show()
        }
        binding.switchAgent.setOnCheckedChangeListener { _, checked ->
            if (suppressAgentSwitchCallback) return@setOnCheckedChangeListener
            if (!checked) {
                setAgentEnabled(false)
            } else {
                val status = projectWorkspace.inspect()
                if (status.ready) {
                    setAgentEnabled(true)
                } else {
                    setAgentEnabled(false)
                    Toast.makeText(this, getString(R.string.agent_requires_workspace), Toast.LENGTH_LONG).show()
                    launchWorkspacePicker(true)
                }
            }
        }
        val initial = projectWorkspace.inspect()
        setAgentEnabled(agentUiPrefs.getBoolean(KEY_AGENT_ENABLED, false) && initial.ready, persist = false)
        refreshWorkspaceUi(initial)
    }

    private fun launchWorkspacePicker(enableAfterSelection: Boolean) {
        pendingEnableAgentAfterPicker = enableAfterSelection
        val picker = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_PREFIX_URI_PERMISSION)
        }
        startActivityForResult(picker, REQUEST_WORKSPACE)
    }

    private fun setAgentEnabled(enabled: Boolean, persist: Boolean = true) {
        agentModeEnabled = enabled
        suppressAgentSwitchCallback = true
        binding.switchAgent.isChecked = enabled
        suppressAgentSwitchCallback = false
        if (persist) agentUiPrefs.edit().putBoolean(KEY_AGENT_ENABLED, enabled).apply()
    }

    private fun refreshWorkspaceUi(status: com.geniex.demo.agent.WorkspaceAccessStatus = projectWorkspace.inspect()) {
        binding.tvWorkspace.text = status.name ?: getString(R.string.workspace_not_selected)
        binding.tvWorkspaceStatus.text = when {
            !status.selected -> getString(R.string.workspace_status_none)
            status.ready -> getString(R.string.workspace_status_ready)
            !status.persistedRead || !status.exists || !status.readable -> getString(R.string.workspace_status_stale)
            else -> getString(R.string.workspace_status_read_only)
        }
        binding.tvWorkspaceStatus.setTextColor(when { status.ready -> Color.rgb(22,130,72); status.selected -> Color.rgb(180,88,32); else -> Color.rgb(120,122,128) })
    }

    private fun handleIncomingShare(incoming: Intent?) {
        if (incoming?.action != Intent.ACTION_SEND && incoming?.action != Intent.ACTION_SEND_MULTIPLE) return
        modelScope.launch {
            val attachments = sharedInputManager.fromIntent(incoming)
            if (attachments.isEmpty()) return@launch
            pendingSharedAttachments.clear()
            pendingSharedAttachments.addAll(attachments)
            val visionFiles = runCatching { sharedInputManager.visionFiles(attachments) }.getOrDefault(emptyList())
            val textContext = sharedInputManager.textContext(attachments)
            val otherNames = attachments.filter { it.kind == SharedAttachment.Kind.OTHER }.joinToString(", ") { it.displayName }
            runOnUiThread {
                savedImageFiles.addAll(visionFiles)
                refreshTopScrollContainer()
                val extra = buildString {
                    if (textContext.isNotBlank()) append(textContext)
                    if (otherNames.isNotBlank()) {
                        if (isNotEmpty()) append("\n\n")
                        append(getString(R.string.shared_files_metadata, otherNames))
                    }
                }
                if (extra.isNotBlank()) {
                    val current = etInput.text?.toString().orEmpty().trim()
                    etInput.setText(listOf(current, extra).filter { it.isNotBlank() }.joinToString("\n\n"))
                    etInput.setSelection(etInput.text.length)
                }
                Toast.makeText(this@MainActivity, getString(R.string.shared_files_summary, attachments.size), Toast.LENGTH_SHORT).show()
                refreshSendButtonState()
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIncomingShare(intent)
    }

    private fun modelInputForAgent(userText: String): String {
        val instructions = mutableListOf<String>()
        if (agentModeEnabled) {
            val status = projectWorkspace.inspect()
            if (status.ready) instructions += "$AGENT_INSTRUCTION\n${getString(R.string.agent_workspace_context, status.name ?: "project")}" else instructions += getString(R.string.agent_requires_workspace)
        }
        if (crossConversationReadAuthorized) {
            instructions += getString(R.string.history_context_turn_enabled)
            instructions += "Use list_projects/list_conversations/read_conversation_context only as needed to satisfy this explicit request. Reading history is read-only; do not merge other conversations into the current conversation unless relevant."
        }
        return if (instructions.isEmpty()) userText else instructions.joinToString("\n\n") + "\n\nUser request:\n" + userText
    }

    private fun explicitlyRequestsHistoricalContext(text: String): Boolean {
        val normalized = text.lowercase(Locale.ROOT)
        val historyTerms = listOf("对话", "会话", "聊天", "历史", "上下文", "项目", "conversation", "chat", "history", "context", "project")
        val actionTerms = listOf("读取", "查看", "参考", "回顾", "调用", "结合", "检索", "搜索", "引用", "read", "look up", "reference", "review", "search", "use context", "recall")
        return historyTerms.any(normalized::contains) && actionTerms.any(normalized::contains)
    }

    private suspend fun continueAgentLoop(initialOutput: String, tools: String) {
        var output = initialOutput
        repeat(MAX_AGENT_STEPS) {
            val calls = AgentToolParser.parse(output)
            if (calls.isEmpty()) return
            val feedback = calls.joinToString("\n\n") { call ->
                val result = agentTools.execute(call.name, call.arguments)
                "<tool_response>\n${call.name}:\n$result\n</tool_response>"
            }
            runOnUiThread {
                messages.add(Message(getString(R.string.agent_tool_activity, calls.size), MessageType.PROFILE))
                reloadRecycleView()
            }
            val next = StringBuilder()
            if (isLoadVlmModel) {
                vlmChatList.add(VlmChatMessage("tool", listOf(VlmContent("text", feedback))))
                vlmWrapper.applyChatTemplate(vlmChatList.toTypedArray(), tools, enableThinking)
                    .onSuccess { templated -> vlmWrapper.generateStreamFlow(templated.formattedText, GenerationConfigSample().toGenerationConfig()).collect { handleResult(next, it) } }
                    .onFailure { error -> runOnUiThread { messages.add(Message(getString(R.string.agent_error, UiErrorLocalizer.message(this@MainActivity, error.message)), MessageType.PROFILE)); reloadRecycleView() } }
            } else if (isLoadLlmModel) {
                chatList.add(ChatMessage("tool", feedback))
                llmWrapper.applyChatTemplate(chatList.toTypedArray(), tools, enableThinking)
                    .onSuccess { templated -> llmWrapper.generateStreamFlow(templated.formattedText, GenerationConfigSample().toGenerationConfig()).collect { handleResult(next, it) } }
                    .onFailure { error -> runOnUiThread { messages.add(Message(getString(R.string.agent_error, UiErrorLocalizer.message(this@MainActivity, error.message)), MessageType.PROFILE)); reloadRecycleView() } }
            }
            output = next.toString()
        }
    }

    private fun resetLoadState() {
        isLoadLlmModel = false
        isLoadVlmModel = false
        // Stale geometry would size preprocessing for the previous model.
        vlmVisionConfig = null
    }

    private fun initView() {
        adapter = ChatAdapter(messages)
        binding.rvChat.adapter = adapter
        setupChatAutoScroll()

        llDownloading = findViewById(R.id.ll_downloading)
        tvDownloadProgress = findViewById(R.id.tv_download_progress)
        pbDownloading = findViewById(R.id.pb_downloading)
        spModelList = findViewById(R.id.sp_model_list)
        spModelList.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long,
                ) {
                    val selected = modelList.getOrNull(position)
                    if (selected == null) {
                        selectModelId = ""
                        return
                    }
                    selectModelId = selected.id
                    getSharedPreferences(ModelLibraryActivity.PREFS, MODE_PRIVATE)
                        .edit().putString(ModelLibraryActivity.KEY_SELECTED_MODEL, selected.modelName).apply()
                    if (!hasLoadedModel()) binding.tvModelState.text = getString(R.string.model_state_selected, selected.displayName)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    selectModelId = ""
                }
            }
        btnDownload = findViewById(R.id.btn_download)
        btnLoadModel = findViewById(R.id.btn_load_model)
        btnUnloadModel = findViewById(R.id.btn_unload_model)
        btnStop = findViewById(R.id.btn_stop)
        etInput = findViewById(R.id.et_input)
        btnAddImage = findViewById(R.id.btn_add_image)

        btnSend = findViewById(R.id.btn_send)
        btnSend.isEnabled = false
        btnLoadModel.isEnabled = modelList.isNotEmpty()
        binding.tvModelState.text = getString(R.string.model_state_none)
        etInput.doAfterTextChanged { refreshSendButtonState() }
        btnClearHistory = findViewById(R.id.btn_clear_history)
        scrollImages = findViewById(R.id.scroll_images)
        topScrollContainer = findViewById(R.id.ll_images_container)
        llLoading = findViewById(R.id.ll_loading)
        vTip = findViewById<View>(R.id.v_tip)
        updateModelSpinnerUi()

        findViewById<Button>(R.id.btn_test).setOnClickListener {
            Thread {
                val exeFile = File(filesDir, "geniex_test_llm")
                val chmodProcess = Runtime.getRuntime().exec("chmod 755 " + exeFile.absolutePath)
                chmodProcess.waitFor()
                Log.d(TAG, "exeFile exe? ${exeFile.canExecute()}")
                Log.d(TAG, "Exe Thread:${Thread.currentThread().name}")
                ExecShell()
                    .executeCommand(
                        arrayOf(
                            "cat",
                            "/sys/devices/soc0/sku",
                        ),
                    ).forEach {
                        Log.d(TAG, "cmd:$it")
                    }
            }.start()
        }

        findViewById<View>(R.id.v_tip).setOnClickListener {
            Toast.makeText(this, getString(R.string.unload_before_load), Toast.LENGTH_SHORT).show()
        }
    }

    private fun refreshCachedModelsIntoSpinner() {
        modelScope.launch {
            val cached = runCatching { ModelManagerWrapper.list() }.getOrDefault(emptyList())
            val dynamic = cached.mapNotNull { name ->
                runCatching {
                    val paths = ModelManagerWrapper.getPaths(name) ?: return@runCatching null
                    catalogModels.firstOrNull { sameModelName(it.modelName, name) } ?: ModelData(
                        id = name,
                        displayName = name.substringAfterLast('/'),
                        modelName = name,
                        type = if (!paths.mmproj_path.isNullOrBlank()) "vlm" else "chat",
                        runtime = paths.runtime_id.ifBlank { "llama_cpp" },
                        hub = "HUGGINGFACE",
                    )
                }.getOrNull()
            }.filterNotNull()
            val seed = if (BuildConfig.TEST_MODEL_SEED) catalogModels.filter { it.id == "Qwen3.5-2B-GGUF" } else emptyList()
            val merged = (dynamic + seed).distinctBy { it.modelName.substringAfterLast('/').lowercase(Locale.ROOT) }
            runOnUiThread {
                modelList = merged
                updateModelSpinnerUi()
            }
        }
    }

    private fun sameModelName(a: String, b: String): Boolean =
        a.equals(b, ignoreCase = true) || a.substringAfterLast('/').equals(b.substringAfterLast('/'), ignoreCase = true)

    private fun updateModelSpinnerUi() {
        val labels = if (modelList.isEmpty()) listOf(getString(R.string.model_selector_empty)) else modelList.map { it.displayName }
        spModelList.adapter = android.widget.ArrayAdapter(this, R.layout.item_model, R.id.tv_model_id, labels).apply {
            setDropDownViewResource(R.layout.item_model)
        }
        spModelList.isEnabled = modelList.isNotEmpty()
        binding.btnQuickInstallModel.visibility = if (modelList.isEmpty()) View.VISIBLE else View.GONE
        btnLoadModel.isEnabled = modelList.isNotEmpty() && !hasLoadedModel()
        btnDownload.isEnabled = modelList.isNotEmpty()
        if (modelList.isEmpty()) {
            selectModelId = ""
            if (!hasLoadedModel()) binding.tvModelState.text = getString(R.string.model_state_none)
            return
        }
        val preferred = getSharedPreferences(ModelLibraryActivity.PREFS, MODE_PRIVATE).getString(ModelLibraryActivity.KEY_SELECTED_MODEL, null)
        val idx = modelList.indexOfFirst { preferred != null && sameModelName(it.modelName, preferred) }.let { if (it >= 0) it else 0 }
        spModelList.setSelection(idx, false)
        val selected = modelList[idx]
        selectModelId = selected.id
        if (!hasLoadedModel()) binding.tvModelState.text = getString(R.string.model_state_selected, selected.displayName)
    }

    private fun selectedModelLabel(): String =
        modelList.firstOrNull { it.id == selectModelId }?.displayName ?: getString(R.string.model_generic)

    override fun onResume() {
        super.onResume()
        if (::spModelList.isInitialized && ::modelList.isInitialized) refreshCachedModelsIntoSpinner()
        if (::sessionStore.isInitialized) {
            refreshProjectList()
            refreshConversationList()
        }
        if (::projectWorkspace.isInitialized) refreshWorkspaceUi()
        if (::imageModeController.isInitialized) imageModeController.onResume()
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        if (!binding.drawerLayout.isDrawerOpen(GravityCompat.START)) {
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    drawerSwipeDownX = event.x
                    drawerSwipeDownY = event.y
                    drawerSwipeTriggered = false
                }
                MotionEvent.ACTION_MOVE -> {
                    if (!drawerSwipeTriggered) {
                        val dx = event.x - drawerSwipeDownX
                        val dy = event.y - drawerSwipeDownY
                        val fromMainArea = drawerSwipeDownX <= resources.displayMetrics.widthPixels * 0.65f
                        if (fromMainArea && dx >= dp(48) && dx > abs(dy) * 1.15f) {
                            drawerSwipeTriggered = true
                            binding.drawerLayout.openDrawer(GravityCompat.START)
                            return true
                        }
                    }
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> drawerSwipeTriggered = false
            }
        }
        return super.dispatchTouchEvent(event)
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (binding.drawerLayout.isDrawerOpen(GravityCompat.START)) binding.drawerLayout.closeDrawer(GravityCompat.START) else super.onBackPressed()
    }

    private fun parseModelList() {
        try {
            val baseJson = assets.open("model_list.json").bufferedReader().use { it.readText() }
            val json = Json { ignoreUnknownKeys = true }
            val catalog = json.decodeFromString<List<ModelData>>(baseJson)
            catalogModels = catalog
            modelList = if (BuildConfig.TEST_MODEL_SEED) catalog.filter { it.id == "Qwen3.5-2B-GGUF" } else emptyList()
        } catch (e: Exception) {
            catalogModels = emptyList()
            modelList = emptyList()
            Log.e(TAG, "parseModelList: $e")
        }
    }

    /**
     * Step 0. Parse the model list and initialise the SDK. Model presence
     * is queried from the Rust model manager, not tracked client-side.
     */
    private fun initData() {
        parseModelList()
        initGenieXSdk()
    }

    /**
     * Step 1. initGenieXSdk environment
     */
    private fun initGenieXSdk() {
        GenieXSdk.getInstance().init(
            this,
            object : GenieXSdk.InitCallback {
                override fun onSuccess() {
                    modelScope.launch {
                        runCatching { TestModelSeeder.ensureSeeded(this@MainActivity) }
                            .onFailure { Log.e(TAG, "debug test model seed failed", it) }
                        refreshCachedModelsIntoSpinner()
                    }
                }

                override fun onFailure(reason: String) {
                    Log.e(TAG, "GenieXSdk init failed: $reason")
                }
            },
        )
    }

    private fun onLoadModelSuccess(tip: String) {
        runOnUiThread {
            Toast
                .makeText(
                    this@MainActivity,
                    tip,
                    Toast.LENGTH_SHORT,
                ).show()
            // change UI
            btnAddImage.visibility = View.GONE
            if (isLoadVlmModel) btnAddImage.visibility = View.VISIBLE
            btnLoadModel.visibility = View.GONE
            btnUnloadModel.visibility = View.VISIBLE
            llLoading.visibility = View.GONE
            btnStop.visibility = View.VISIBLE
            binding.tvModelState.text = getString(R.string.model_state_loaded, selectedModelLabel())
            refreshSendButtonState()
        }
    }

    private fun onLoadModelFailed(tip: String) {
        runOnUiThread {
            vTip.visibility = View.GONE
            Toast.makeText(this@MainActivity, tip, Toast.LENGTH_SHORT).show()
            // change UI
            btnAddImage.visibility = View.GONE
            btnLoadModel.visibility = View.VISIBLE
            btnUnloadModel.visibility = View.GONE
            llLoading.visibility = View.GONE
            binding.tvModelState.text = getString(R.string.model_state_failed)
        }
    }

    private fun hasLoadedModel(): Boolean = isLoadLlmModel || isLoadVlmModel

    /**
     * Send is enabled only when (a) a model is loaded, (b) no inference
     * is in flight, and (c) there is something to send — text or an
     * attached image (VLM only).
     */
    private fun refreshSendButtonState() {
        runOnUiThread {
            val hasText = etInput.text?.isNotBlank() == true
            val hasAttachment = savedImageFiles.isNotEmpty()
            btnSend.isEnabled = hasLoadedModel() && !isGenerating && (hasText || hasAttachment)
        }
    }

    /**
     * Checks the Rust model manager's cache for [modelData]. Uses
     * `getPaths`, which canonicalises the name (so `ai-hub-models/<repo>`
     * and `qualcomm/<repo>` map to the same on-disk entry) and returns
     * null while the pull is still in `.inflight/`.
     */
    private suspend fun isModelDownloaded(modelData: ModelData): Boolean = ModelManagerWrapper.getPaths(modelData.modelName) != null

    private fun loadModel(
        selectModelData: ModelData,
        modelDataPluginId: String,
        nGpuLayers: Int,
        deviceId: String? = null,
    ) {
        modelScope.launch {
            resetLoadState()
            val paths = ModelManagerWrapper.getPaths(selectModelData.modelName)
            if (paths == null) {
                onLoadModelFailed(getString(R.string.model_paths_unavailable))
                return@launch
            }
            // Manifest-written runtime_id wins when present; fall back to
            // the user's UI selection for GGUF models that skip the manifest.
            val pluginId = paths.runtime_id.ifEmpty { modelDataPluginId }
            val resolvedDeviceId = deviceId
            when (selectModelData.type) {
                "chat", "llm" -> {
                    // QAIRT rejects non-zero n_ctx / n_gpu_layers (both fixed at compile
                    // time in the AI Hub bundle) — and the Kotlin ModelConfig defaults
                    // are non-zero, so zero them explicitly for the qairt path.
                    val isQairt = pluginId == "qairt"
                    val conf =
                        if (isQairt) {
                            ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = enableThinking)
                        } else {
                            ModelConfig(
                                nCtx = 1024,
                                nGpuLayers = nGpuLayers,
                                enable_thinking = enableThinking,
                            )
                        }
                    LlmWrapper
                        .builder()
                        .llmCreateInput(
                            LlmCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                tokenizer_path = paths.tokenizer_path,
                                config = conf,
                                runtime_id = pluginId,
                                compute_unit = resolvedDeviceId ?: ComputeUnitValue.NPU.value,
                            ),
                        ).build()
                        .onSuccess { wrapper ->
                            isLoadLlmModel = true
                            llmWrapper = wrapper
                            onLoadModelSuccess(getString(R.string.model_loaded))
                        }.onFailure { error ->
                            onLoadModelFailed(UiErrorLocalizer.message(this@MainActivity, error.message))
                        }
                }

                "multimodal", "vlm" -> {
                    val isNpuVlm = pluginId == "qairt"
                    // Size image preprocessing from the tower this model actually
                    // ships, not from a constant: Qwen3.5-VL is 768/16 (576
                    // tokens) but Qwen2.5-VL is 560/14 (1600), so one hardcoded
                    // number mis-sizes every other model in the catalog.
                    vlmVisionConfig =
                        paths.mmproj_path?.takeIf { it.isNotEmpty() }?.let { GgufVisionReader.read(File(it)) }
                    vlmVisionConfig?.let {
                        Log.d(
                            TAG,
                            "vision tower: ${it.imageSize}px, patch ${it.patchSize}, " +
                                "merge ${it.spatialMergeSize} -> ${it.tokenCount} image tokens",
                        )
                    } ?: Log.w(TAG, "no vision config from mmproj; preprocessing at $FALLBACK_VLM_IMAGE_SIZE")
                    val config =
                        if (isNpuVlm) {
                            // QAIRT rejects non-zero n_ctx / n_gpu_layers for VLM too.
                            ModelConfig(nCtx = 0, nGpuLayers = 0, nThreads = 8, enable_thinking = enableThinking)
                        } else {
                            ModelConfig(
                                // One image costs tokenCount tokens (576 on
                                // Qwen3.5-VL, 1600 on Qwen2.5-VL). nCtx = 1024
                                // left too little room for the prompt plus a
                                // reply, and a second image turn died inside
                                // mtmd_tokenize with "failed to initialize
                                // batch". Leave room for an image, its answer and
                                // a follow-up turn.
                                nCtx = vlmContextSize(vlmVisionConfig),
                                nThreads = 4,
                                nBatch = 1,
                                nUBatch = 1,
                                nGpuLayers = nGpuLayers,
                                enable_thinking = enableThinking,
                            )
                        }
                    VlmWrapper
                        .builder()
                        .vlmCreateInput(
                            VlmCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                mmproj_path = paths.mmproj_path,
                                config = config,
                                runtime_id = pluginId,
                                compute_unit = resolvedDeviceId ?: "HTP0",
                            ),
                        ).build()
                        .onSuccess {
                            isLoadVlmModel = true
                            vlmWrapper = it
                            onLoadModelSuccess(getString(R.string.model_loaded))
                        }.onFailure { error ->
                            onLoadModelFailed(UiErrorLocalizer.message(this@MainActivity, error.message))
                        }
                }

                else -> {
                    onLoadModelFailed(getString(R.string.model_type_error))
                }
            }
        }
    }

    private fun downloadModel(selectModelData: ModelData) {
        if (hasLoadedModel()) {
            Toast.makeText(this@MainActivity, getString(R.string.unload_before_download), Toast.LENGTH_SHORT).show()
            return
        }
        if (downloadJob?.isActive == true) {
            Toast
                .makeText(
                    this@MainActivity,
                    getString(R.string.download_already_running, downloadingModelData?.displayName ?: getString(R.string.model_generic)),
                    Toast.LENGTH_SHORT,
                ).show()
            return
        }

        downloadingModelData = selectModelData
        llDownloading.visibility = View.VISIBLE
        tvDownloadProgress.text = "0%"

        val hub =
            runCatching { HubSource.valueOf(selectModelData.hub ?: "AUTO") }
                .getOrDefault(HubSource.AUTO)
        // AI Hub pulls route through chipset-matched assets. The Rust side
        // can auto-detect the host only on Windows-on-Snapdragon, so on
        // Android we must pass an explicit chipset for anything that ends
        // up on the AI Hub path — whether hub is AIHUB or AUTO + ai-hub-models/*
        // (or its canonical alias qualcomm/*).
        val name = selectModelData.modelName
        val isAiHubName =
            name.startsWith("ai-hub-models/", ignoreCase = true) ||
                name.startsWith("qualcomm/", ignoreCase = true)
        val willUseAiHub =
            hub == HubSource.AIHUB ||
                (hub == HubSource.AUTO && isAiHubName)
        if (willUseAiHub && selectModelData.chipset.isNullOrBlank()) {
            llDownloading.visibility = View.GONE
            Toast.makeText(this@MainActivity, getString(R.string.aihub_chipset_required), Toast.LENGTH_SHORT).show()
            return
        }
        val input =
            ModelPullInput(
                model_name = selectModelData.modelName,
                precision = selectModelData.quant,
                hub = hub,
                chipset = selectModelData.chipset,
                display_name = selectModelData.aiHubDisplayName,
            )

        val wakeLock =
            (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "geniex:model_download")
        wakeLock.acquire()
        downloadJob =
            modelScope.launch {
                try {
                    // Short-circuit if already cached — the manager filters .inflight/
                    // models out of list(), so this only matches a complete pull.
                    if (isModelDownloaded(selectModelData)) {
                        runOnUiThread {
                            llDownloading.visibility = View.GONE
                            Toast.makeText(this@MainActivity, getString(R.string.model_already_downloaded), Toast.LENGTH_SHORT).show()
                        }
                        return@launch
                    }

                    ModelManagerWrapper.pullFlow(input).collect { event ->
                        when (event) {
                            is ModelManagerWrapper.PullEvent.Progress -> {
                                val total = event.files.sumOf { if (it.total_bytes > 0) it.total_bytes else 0L }
                                val done = event.files.sumOf { it.downloaded_bytes }
                                val percent = if (total > 0) ((done * 100) / total).toInt() else 0
                                runOnUiThread { tvDownloadProgress.text = "$percent%" }
                            }

                            is ModelManagerWrapper.PullEvent.Completed -> {
                                runOnUiThread {
                                    llDownloading.visibility = View.GONE
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            getString(R.string.model_downloaded, selectModelData.displayName),
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                }
                            }

                            is ModelManagerWrapper.PullEvent.Error -> {
                                Log.e(TAG, "pull failed rc=${event.code}: ${event.message}")
                                runOnUiThread {
                                    llDownloading.visibility = View.GONE
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            getString(R.string.model_download_failed),
                                            Toast.LENGTH_LONG,
                                        ).show()
                                }
                            }
                        }
                    }
                } finally {
                    if (wakeLock.isHeld) wakeLock.release()
                }
            }
    }

    private fun releaseChatRuntimeForImage(after: () -> Unit) {
        if (isGenerating) {
            modelScope.launch {
                runCatching {
                    if (isLoadVlmModel) vlmWrapper.stopStream() else if (isLoadLlmModel) llmWrapper.stopStream()
                }
                isGenerating = false
                runOnUiThread { refreshSendButtonState() }
                unloadLoadedModel(showToast = false, after = after)
            }
            return
        }
        unloadLoadedModel(showToast = false, after = after)
    }

    private fun unloadLoadedModel(showToast: Boolean, after: () -> Unit = {}) {
        if (!hasLoadedModel()) {
            runOnUiThread { after() }
            return
        }
        modelScope.launch {
            val result = runCatching {
                if (isLoadVlmModel) {
                    runCatching { vlmWrapper.stopStream() }
                    vlmWrapper.destroy()
                    0
                } else if (isLoadLlmModel) {
                    runCatching { llmWrapper.stopStream() }
                    llmWrapper.destroy()
                    0
                } else {
                    0
                }
            }.getOrElse {
                Log.e(TAG, "model unload failed", it)
                -1
            }
            resetLoadState()
            runOnUiThread {
                vTip.visibility = View.GONE
                btnLoadModel.visibility = View.VISIBLE
                btnUnloadModel.visibility = View.GONE
                btnStop.visibility = View.GONE
                btnAddImage.visibility = View.GONE
                binding.tvModelState.text = getString(R.string.model_state_none)
                clearImages()
                rebuildTemplateHistories()
                refreshSendButtonState()
                if (showToast) {
                    Toast.makeText(
                        this@MainActivity,
                        if (result == 0) getString(R.string.model_unloaded) else getString(R.string.unload_failed_code, result),
                        Toast.LENGTH_SHORT,
                    ).show()
                }
                after()
            }
        }
    }

    private fun setListeners() {
        binding.drawerLayout.setDrawerLockMode(androidx.drawerlayout.widget.DrawerLayout.LOCK_MODE_UNLOCKED)
        binding.btnOpenDrawer.setOnClickListener { binding.drawerLayout.openDrawer(GravityCompat.START) }
        binding.btnCloseDrawer.setOnClickListener { binding.drawerLayout.closeDrawer(GravityCompat.START) }
        binding.btnModelLibrary.setOnClickListener {
            binding.drawerLayout.closeDrawer(GravityCompat.START)
            startActivity(Intent(this, ModelLibraryActivity::class.java))
        }
        binding.btnModelUpdate.setOnClickListener {
            binding.drawerLayout.closeDrawer(GravityCompat.START)
            startActivity(Intent(this, ModelUpdateActivity::class.java))
        }
        binding.btnQuickInstallModel.setOnClickListener {
            binding.drawerLayout.closeDrawer(GravityCompat.START)
            startActivity(Intent(this, ModelLibraryActivity::class.java).putExtra(ModelLibraryActivity.EXTRA_AUTO_INSTALL_REPO, RecommendedModels.primary.repoId))
        }
        btnAddImage.setOnClickListener { showPopupMenu(btnAddImage) }

        btnClearHistory.setOnClickListener {
            clearHistory()
        }
        /*
         * Step 3. download model. Cancelling the coroutine triggers the
         * flow's awaitClose which flips the Rust progress callback to
         * return false — partial files stay on disk for a resumed pull.
         * Use the Retry button to kick off a fresh pull that resumes.
         */
        binding.btnCancelDownload.setOnClickListener {
            downloadJob?.cancel()
            downloadJob = null
            tvDownloadProgress.text = "0%"
            binding.llDownloading.visibility = View.GONE
        }
        binding.btnRetryDownload.setOnClickListener {
            downloadJob?.cancel()
            downloadJob = null
            downloadingModelData?.let { downloadModel(it) }
        }
        btnDownload.setOnClickListener {
            if (downloadJob?.isActive == true) {
                if (downloadingModelData?.id == selectModelId) {
                    binding.llDownloading.visibility = View.VISIBLE
                } else {
                    Toast
                        .makeText(
                            this@MainActivity,
                            getString(R.string.download_in_progress, downloadingModelData?.displayName ?: getString(R.string.model_generic)),
                            Toast.LENGTH_SHORT,
                        ).show()
                }
                return@setOnClickListener
            }
            val selectModelData = modelList.first { it.id == selectModelId }
            downloadModel(selectModelData)
        }
        /*
         * Step 4. load model
         */
        btnLoadModel.setOnClickListener {
            val selectModelData = modelList.first { it.id == selectModelId }
            Log.d(TAG, "current select model data:$selectModelData")
            if (hasLoadedModel()) {
                Toast.makeText(this@MainActivity, getString(R.string.unload_before_load), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Availability is checked against the manager's cache — a pull
            // that was cancelled mid-flight is not listed until it completes.
            modelScope.launch {
                if (!isModelDownloaded(selectModelData)) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, getString(R.string.model_not_downloaded_tap), Toast.LENGTH_LONG).show()
                    }
                    return@launch
                }
                runOnUiThread { startLoadModel(selectModelData) }
            }
        }

        /*
         * Step 5. send message
         */
        btnSend.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast
                    .makeText(this@MainActivity, getString(R.string.model_not_loaded), Toast.LENGTH_SHORT)
                    .show()
                return@setOnClickListener
            }
            // Guard against re-entry: a second click while a previous
            // generate() is still running would race on the native handle
            // and crash the app.
            if (isGenerating) return@setOnClickListener
            isGenerating = true
            refreshSendButtonState()

            if (savedImageFiles.isNotEmpty()) {
                messages.add(Message("", MessageType.IMAGES, savedImageFiles.map { it }))
                reloadRecycleView()
            }

            val inputString = etInput.text.trim().toString()
            etInput.setText("")
            etInput.clearFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(etInput.windowToken, 0)

            if (inputString.isNotEmpty()) {
                messages.add(Message(inputString, MessageType.USER))
                reloadRecycleView()
            }
            persistCurrentConversation()

            showLoadingIndicator()

            val agentStatus = if (agentModeEnabled) projectWorkspace.inspect() else null
            if (agentModeEnabled && agentStatus?.ready != true) {
                setAgentEnabled(false)
                refreshWorkspaceUi(agentStatus ?: projectWorkspace.inspect())
                Toast.makeText(this, getString(R.string.agent_requires_workspace), Toast.LENGTH_LONG).show()
            }
            crossConversationReadAuthorized = explicitlyRequestsHistoricalContext(inputString)
            val fileToolsEnabled = agentModeEnabled && agentStatus?.ready == true
            val historyToolsEnabled = crossConversationReadAuthorized
            val tools: String? = if (fileToolsEnabled || historyToolsEnabled) agentTools.definitions(fileToolsEnabled, historyToolsEnabled) else null

            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, getString(R.string.model_not_loaded), Toast.LENGTH_SHORT).show()
                isGenerating = false
                refreshSendButtonState()
                return@setOnClickListener
            }

            modelScope.launch {
                try {
                    val modelInputString = modelInputForAgent(inputString)
                    val selectModelData = modelList.first { it.id == selectModelId }
                    val isNpu = ModelManagerWrapper.getPaths(selectModelData.modelName)?.runtime_id == "qairt"
                    Log.d(TAG, "isNpu: $isNpu")

                    val sb = StringBuilder()
                    if (isLoadVlmModel) {
                        val contents =
                            savedImageFiles
                                .map {
                                    VlmContent("image", it.absolutePath)
                                }.toMutableList()
                        contents.add(VlmContent("text", modelInputString))
                        clearImages()
                        val sendMsg = VlmChatMessage(role = "user", contents = contents)
                        vlmChatList.add(sendMsg)

                        Log.d(TAG, "before apply chat template:$vlmChatList")
                        vlmWrapper
                            .applyChatTemplate(vlmChatList.toTypedArray(), tools, enableThinking)
                            .onSuccess { result ->
                                Log.d(TAG, "vlm chat template:${result.formattedText}")
                                val baseConfig =
                                    GenerationConfigSample().toGenerationConfig()
                                // Only inject the current turn's media: SDK tokenizes
                                // incrementally, so re-passing history bitmaps breaks
                                // mtmd_tokenize (markers/bitmaps mismatch).
                                val configWithMedia =
                                    vlmWrapper.injectMediaPathsToConfig(
                                        arrayOf(sendMsg),
                                        baseConfig,
                                    )

                                Log.d(TAG, "Config has ${configWithMedia.imageCount} images")

                                vlmWrapper
                                    .generateStreamFlow(
                                        result.formattedText,
                                        configWithMedia,
                                    ).collect { handleResult(sb, it) }
                            }.onFailure {
                                runOnUiThread {
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            UiErrorLocalizer.message(this@MainActivity, it.message),
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                }
                            }
                    } else {
                        chatList.add(ChatMessage(role = "user", modelInputString))
                        // Apply chat template and generate
                        llmWrapper
                            .applyChatTemplate(
                                chatList.toTypedArray(),
                                tools,
                                enableThinking,
                            ).onSuccess { templateOutput ->
                                Log.d(TAG, "chat template:${templateOutput.formattedText}")
                                llmWrapper
                                    .generateStreamFlow(
                                        templateOutput.formattedText,
                                        GenerationConfigSample().toGenerationConfig(),
                                    ).collect { streamResult ->
                                        handleResult(sb, streamResult)
                                    }
                            }.onFailure { error ->
                                runOnUiThread {
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            UiErrorLocalizer.message(this@MainActivity, error.message),
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                }
                            }
                    }

                    if (tools != null) {
                        continueAgentLoop(sb.toString(), tools)
                    }
                    clearImages()
                } finally {
                    removeLoadingIndicator()
                    isGenerating = false
                    refreshSendButtonState()
                    crossConversationReadAuthorized = false
                }
            }
        }

        /*
         * Step 6. others
         */
        btnUnloadModel.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, getString(R.string.model_not_loaded), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            unloadLoadedModel(showToast = true)
        }
        btnStop.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast
                    .makeText(
                        this@MainActivity,
                        getString(R.string.model_not_loaded),
                        Toast.LENGTH_SHORT,
                    ).show()
                return@setOnClickListener
            }
            // Stop streaming
            modelScope.launch {
                if (isLoadVlmModel) {
                    vlmWrapper.stopStream()
                } else if (isLoadLlmModel) {
                    llmWrapper.stopStream()
                }
            }
        }
    }

    private fun startLoadModel(selectModelData: ModelData) {
        vTip.visibility = View.VISIBLE
        llLoading.visibility = View.VISIBLE
        binding.tvModelState.text = getString(R.string.model_state_loading)

        val supportPluginIds = selectModelData.getSupportPluginIds()
        Log.d(TAG, "support plugin_id:$supportPluginIds")
        var modelDataPluginId = "llama_cpp"
        var nGpuLayers = 0
        if (supportPluginIds.size > 1) {
            val dialogBinding = DialogSelectPluginIdBinding.inflate(layoutInflater)
            val isGgufLlmModel =
                !selectModelData.isNpuModel() &&
                    (selectModelData.type == "chat" || selectModelData.type == "llm")
            supportPluginIds.forEach {
                when (it) {
                    "cpu" -> {
                        dialogBinding.rbCpu.visibility = View.VISIBLE
                        dialogBinding.rbCpu.isChecked = true
                    }

                    "gpu" -> {
                        dialogBinding.rbGpu.visibility = View.VISIBLE
                    }

                    "npu" -> {
                        dialogBinding.rbNpu.visibility = View.VISIBLE
                        dialogBinding.rbNpu.isChecked = true
                    }
                }
            }
            if (isGgufLlmModel) {
                dialogBinding.rbNpu.visibility = View.VISIBLE
            }
            dialogBinding.rgSelectPluginId.setOnCheckedChangeListener { _, checkedId ->
                dialogBinding.llGpuLayers.visibility =
                    if (checkedId == R.id.rb_gpu) View.VISIBLE else View.GONE
            }

            val dialogOnClickListener =
                object : CustomDialogInterface.OnClickListener() {
                    override fun onClick(
                        dialog: DialogInterface?,
                        which: Int,
                    ) {
                        nGpuLayers = 0
                        var ggufLlmDeviceId: String? = null
                        val checkedId = dialogBinding.rgSelectPluginId.checkedRadioButtonId
                        if (checkedId == R.id.rb_gpu) {
                            if (dialogBinding.llGpuLayers.visibility == View.VISIBLE) {
                                nGpuLayers =
                                    dialogBinding.etGpuLayers.text
                                        .toString()
                                        .toInt()
                                if (nGpuLayers == 0) {
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            getString(R.string.gpu_layers_min),
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                    return
                                }
                            }
                            ggufLlmDeviceId = ComputeUnitValue.GPU.value
                        } else if (checkedId == R.id.rb_npu) {
                            nGpuLayers = 999
                            ggufLlmDeviceId = ComputeUnitValue.NPU.value
                        } else if (checkedId == R.id.rb_cpu) {
                            ggufLlmDeviceId = ComputeUnitValue.CPU.value
                        }
                        when (which) {
                            DialogInterface.BUTTON_POSITIVE -> {
                                dialog?.dismiss()
                                loadModel(selectModelData, modelDataPluginId, nGpuLayers, ggufLlmDeviceId)
                            }

                            DialogInterface.BUTTON_NEGATIVE -> {
                                llLoading.visibility = View.INVISIBLE
                                vTip.visibility = View.GONE
                            }
                        }
                    }
                }
            val alertDialog =
                AlertDialog
                    .Builder(this)
                    .setView(dialogBinding.root)
                    .setNegativeButton(android.R.string.cancel, dialogOnClickListener)
                    .setPositiveButton(android.R.string.ok, dialogOnClickListener)
                    .setCancelable(false)
                    .create()
            alertDialog.show()
            dialogOnClickListener.resetPositiveButton(alertDialog)
        } else {
            if ("npu" == supportPluginIds[0]) {
                modelDataPluginId = "npu"
            }
            loadModel(selectModelData, modelDataPluginId, nGpuLayers)
        }
    }

    fun handleResult(
        sb: StringBuilder,
        streamResult: LlmStreamResult,
    ) {
        when (streamResult) {
            is LlmStreamResult.Token -> {
                removeLoadingIndicator()
                runOnUiThread {
                    sb.append(streamResult.text)
                    val lastMsg = Message(sb.toString(), MessageType.ASSISTANT)
                    if (messages.isEmpty() || messages.last().type != MessageType.ASSISTANT) {
                        messages.add(lastMsg)
                    } else {
                        messages[messages.lastIndex] = lastMsg
                    }
                    adapter.notifyDataSetChanged()
                    scrollChatToBottomIfFollowing()
                }
                Log.d(TAG, "Token: ${streamResult.text}")
            }

            is LlmStreamResult.Completed -> {
                removeLoadingIndicator()
                if (isLoadVlmModel) {
                    vlmChatList.add(
                        VlmChatMessage(
                            "assistant",
                            listOf(VlmContent("text", sb.toString())),
                        ),
                    )
                } else {
                    chatList.add(ChatMessage("assistant", sb.toString()))
                }

                runOnUiThread {
                    val content = sb.toString()
                    val assistant = Message(content, MessageType.ASSISTANT)
                    if (messages.isEmpty() || messages.last().type != MessageType.ASSISTANT) messages.add(assistant) else messages[messages.lastIndex] = assistant

                    val ttft = String.format(Locale.US, "%.2f", streamResult.profile.ttftMs)
                    val promptTokens = streamResult.profile.promptTokens
                    val prefillSpeed =
                        String.format(Locale.US, "%.2f", streamResult.profile.prefillSpeed)

                    val generatedTokens = streamResult.profile.generatedTokens
                    val decodingSpeed =
                        String.format(Locale.US, "%.2f", streamResult.profile.decodingSpeed)

                    val profileData =
                        getString(R.string.profile_format, ttft, promptTokens, prefillSpeed, generatedTokens, decodingSpeed)
                    messages.add(
                        Message(
                            profileData,
                            MessageType.PROFILE,
                        ),
                    )
                    reloadRecycleView()
                    persistCurrentConversation()
                }
                Log.d(TAG, "Completed: ${streamResult.profile}")
            }

            is LlmStreamResult.Error -> {
                removeLoadingIndicator()
                runOnUiThread {
                    val reason = streamResult.throwable.message ?: streamResult.throwable.toString()
                    messages.add(Message(UiErrorLocalizer.message(this, reason), MessageType.PROFILE))
                    reloadRecycleView()
                    persistCurrentConversation()
                }
                Log.d(TAG, "Error: $streamResult")
            }
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, null)
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*")
        startActivityForResult(intent, 1)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 0) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery()
            } else {
                Toast.makeText(this, getString(R.string.permission_denied), Toast.LENGTH_SHORT).show()
            }
        } else if (requestCode == 2001) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?,
    ) {
        super.onActivityResult(requestCode, resultCode, data)

        if (::imageModeController.isInitialized && imageModeController.onActivityResult(requestCode, resultCode, data)) return

        if (requestCode == REQUEST_WORKSPACE) {
            val uri = data?.data
            if (resultCode == Activity.RESULT_OK && uri != null) {
                val grantFlags = data.flags
                modelScope.launch {
                    val result = runCatching {
                        projectWorkspace.setRoot(uri, grantFlags)
                        projectWorkspace.probeAccess()
                    }
                    runOnUiThread {
                        result.onSuccess { status ->
                            refreshWorkspaceUi(status)
                            if (status.ready) {
                                if (pendingEnableAgentAfterPicker) setAgentEnabled(true)
                                Toast.makeText(this@MainActivity, getString(R.string.workspace_access_ok), Toast.LENGTH_SHORT).show()
                            } else {
                                setAgentEnabled(false)
                                Toast.makeText(this@MainActivity, getString(R.string.workspace_access_failed, status.detail ?: getString(R.string.workspace_permission_failed)), Toast.LENGTH_LONG).show()
                            }
                        }.onFailure { error ->
                            setAgentEnabled(false)
                            refreshWorkspaceUi()
                            Toast.makeText(this@MainActivity, getString(R.string.workspace_access_failed, error.message ?: getString(R.string.workspace_permission_failed)), Toast.LENGTH_LONG).show()
                        }
                        pendingEnableAgentAfterPicker = false
                    }
                }
            } else {
                pendingEnableAgentAfterPicker = false
            }
            return
        }

        var bitmap: Bitmap? = null
        if (requestCode == 1) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                val inputStream = contentResolver.openInputStream(data.data!!)
                bitmap = BitmapFactory.decodeStream(inputStream)
            }
        } else if (requestCode == 1001 && resultCode == Activity.RESULT_OK) {
            photoFile?.let {
                bitmap = BitmapFactory.decodeFile(it.absolutePath)
            }
        }

        bitmap?.let {
            try {
                val file = File(filesDir, "chat_${System.currentTimeMillis()}.jpg")
                val success = saveBitmapToFile(it, file)
                if (success) {
                    Log.d(TAG, "Save success: ${file.absolutePath}")
                    savedImageFiles.add(file)
                    refreshTopScrollContainer()
                } else {
                    Toast.makeText(this, getString(R.string.image_save_failed), Toast.LENGTH_SHORT).show()
                }
            } catch (e: FileNotFoundException) {
                Log.e(TAG, "save image failed", e)
            }
        }
    }

    private fun saveBitmapToFile(
        bitmap: Bitmap,
        file: File,
    ): Boolean =
        try {
            val tempDir = File(this.filesDir, "tmp").apply { if (!exists()) mkdirs() }

            val tempFile =
                File(
                    tempDir,
                    "tmp_${System.currentTimeMillis()}.jpg",
                )
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }

            // Crop straight from the full-size temp file. Pre-downscaling on the
            // *longest* edge first would leave the shorter edge under the target
            // (e.g. 448x355), forcing squareCrop to upscale it back — two lossy
            // resamples for a softer result. squareCrop samples down internally.
            ImgUtil.squareCrop(
                imageFile = tempFile,
                outFile = file,
                size = vlmVisionConfig?.imageSize ?: FALLBACK_VLM_IMAGE_SIZE,
            )
            true
        } catch (e: Exception) {
            Log.e(TAG, "saveBitmapToFile failed", e)
            false
        }

    private fun clearHistory() {
        if (isLoadLlmModel) {
            chatList.clear()
            modelScope.launch {
                llmWrapper.reset()
            }
        }
        if (isLoadVlmModel) {
            vlmChatList.clear()
            modelScope.launch {
                vlmWrapper.reset()
            }
        }
        messages.clear()
        clearImages()
        rebuildTemplateHistories()
        reloadRecycleView()
        persistCurrentConversation()
    }

    private var popupWindow: PopupWindow? = null

    private fun showPopupMenu(anchorView: View) {
        if (popupWindow?.isShowing == true) {
            popupWindow?.dismiss()
            return
        }

        val popupView = LayoutInflater.from(this).inflate(R.layout.menu_layout, null)

        popupWindow =
            PopupWindow(
                popupView,
                anchorView.width * 2,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                true,
            )

        popupWindow?.isOutsideTouchable = true
        popupWindow?.elevation = 10f

        val btnCamera = popupView.findViewById<Button>(R.id.btn_camera)
        val btnPhoto = popupView.findViewById<Button>(R.id.btn_photo)

        btnCamera.setOnClickListener {
            popupWindow?.dismiss()
            checkAndOpenCamera()
        }
        btnPhoto.setOnClickListener {
            popupWindow?.dismiss()
            openGallery()
        }

        popupView.measure(
            View.MeasureSpec.UNSPECIFIED,
            View.MeasureSpec.UNSPECIFIED,
        )
        val popupHeight = popupView.measuredHeight
        popupWindow?.showAsDropDown(anchorView, 0, -anchorView.height - popupHeight)
    }

    private var photoUri: Uri? = null
    private var photoFile: File? = null

    private fun checkAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                2001,
            )
        } else {
            openCamera()
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile =
            File(
                getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                "photo_${System.currentTimeMillis()}.jpg",
            )
        photoUri =
            FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                photoFile!!,
            )

        intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
        intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        startActivityForResult(intent, 1001)
    }

    private fun clearImages() {
        savedImageFiles.clear()
        refreshTopScrollContainer()
    }

    private fun refreshTopScrollContainer() {
        refreshSendButtonState()
        runOnUiThread {
            topScrollContainer.removeAllViews()
            if (savedImageFiles.isEmpty()) {
                scrollImages.visibility = View.GONE
                return@runOnUiThread
            }

            scrollImages.visibility = View.VISIBLE

            for (file in savedImageFiles) {
                val itemView =
                    LayoutInflater
                        .from(this)
                        .inflate(R.layout.item_image_scroll, topScrollContainer, false)
                val ivImage = itemView.findViewById<ImageView>(R.id.iv_image)
                val btnRemove = itemView.findViewById<ImageButton>(R.id.btn_remove)

                ivImage.setImageURI(Uri.fromFile(file))

                btnRemove.setOnClickListener {
                    savedImageFiles.remove(file)
                    refreshTopScrollContainer()
                }
                topScrollContainer.addView(itemView)
            }
        }
    }

    private fun reloadRecycleView() {
        adapter.notifyDataSetChanged()
        scrollChatToBottomIfFollowing()
    }

    private fun setupChatAutoScroll() {
        binding.rvChat.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING) chatUserDragging = true
                if (newState == RecyclerView.SCROLL_STATE_IDLE && chatUserDragging) {
                    followLatestMessages = !recyclerView.canScrollVertically(1)
                    chatUserDragging = false
                }
            }

            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                if (chatUserDragging) followLatestMessages = !recyclerView.canScrollVertically(1)
            }
        })
    }

    private fun scrollChatToBottomIfFollowing(force: Boolean = false) {
        if (messages.isEmpty()) return
        binding.rvChat.post {
            if (force || followLatestMessages) binding.rvChat.scrollToPosition(messages.lastIndex)
        }
    }

    private fun showLoadingIndicator() {
        runOnUiThread {
            if (loadingMessageIndex >= 0) return@runOnUiThread
            messages.add(Message("", MessageType.LOADING))
            loadingMessageIndex = messages.size - 1
            reloadRecycleView()
        }
    }

    private fun removeLoadingIndicator() {
        runOnUiThread {
            val idx = loadingMessageIndex
            if (idx < 0 || idx >= messages.size) {
                loadingMessageIndex = -1
                return@runOnUiThread
            }
            if (messages[idx].type == MessageType.LOADING) {
                messages.removeAt(idx)
                adapter.notifyItemRemoved(idx)
                scrollChatToBottomIfFollowing()
            }
            loadingMessageIndex = -1
        }
    }

    override fun onStop() {
        persistCurrentConversation()
        super.onStop()
    }

    override fun onDestroy() {
        if (::imageModeController.isInitialized) imageModeController.dispose()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "GenieXDemo"
        private const val REQUEST_WORKSPACE = 3001
        private const val MAX_AGENT_STEPS = 6
        private const val KEY_AGENT_ENABLED = "agent_enabled"
        private const val AGENT_INSTRUCTION = "You are Rin NPU Agent running fully on this Android device. The current logical project and current conversation are isolated from other projects/conversations by default. When file Agent mode is enabled, use file tools only inside the current project's authorized workspace. Never claim a file/folder was changed unless a tool result confirms it. For PowerPoint .pptx output, ALWAYS use create_presentation. Persistent history tools may appear only on turns where the user explicitly requested reading/reference to another project or conversation; use them read-only and only for that request."

        /**
         * Square edge length used for image preprocessing when the mmproj GGUF
         * does not declare one. Only a fallback — the real value is read per
         * model by [GgufVisionReader], since feeding a tower a smaller square
         * than it was trained on silently discards detail.
         */
        private const val FALLBACK_VLM_IMAGE_SIZE = 448

        /** Room for an image, its answer, and a follow-up turn, over the image cost. */
        private const val VLM_CTX_HEADROOM = 2048

        /** nCtx must be at least this regardless of image cost. */
        private const val VLM_MIN_CTX = 4096

        /**
         * Context size that fits one image of [vision]'s token cost plus room to
         * answer and ask again. Rounded up to a power of two, which is what
         * llama.cpp KV-cache allocation prefers.
         */
        private fun vlmContextSize(vision: GgufVisionConfig?): Int {
            val needed = (vision?.tokenCount ?: 0) + VLM_CTX_HEADROOM
            var ctx = VLM_MIN_CTX
            while (ctx < needed) ctx *= 2
            return ctx
        }
    }
}
